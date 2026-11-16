"""The publish gate. Code checks first (free, exact), then one DeepSeek review.

Code:
  - the bold prediction line names the model's pick and method
  - every number outside a quote is in the fact sheet (or a form of one); numbers that are
    not, and that the reporting contains, are rejected outright: a stat from the news
  - every quoted passage is word for word in its claim's source story, and the speaker it
    belongs to is named next to it
  - embeds are ones the reporting found
Review:
  - the article argues for the pick; narrative statements are supported by the claims;
    any remaining unmatched numbers are derivable from the stats
"""

from __future__ import annotations

import re

from app.services.ufc.newsroom import llm
from app.services.ufc.newsroom.claims import _norm, quote_in
from app.services.ufc.newsroom.facts import FactSheet
from app.services.ufc.newsroom.schemas import CheckReport, Claim

FENCE = re.compile(r"```([\w-]*)\n([\s\S]*?)```")
#: A quotation: on one line, starting and ending on a word, so a closing mark in one
#: paragraph is never paired with the opening mark of the next.
QUOTED = re.compile(r"[\"“](?=\S)([^\"“”\n]{12,}?)(?<=\S)[\"”]")
PRED = re.compile(r"\*\*Prediction:\s*(.+?)\s+by\s+(.+?)\.?\*\*", re.I)
METHOD_WORDS = {
    "KO/TKO": ("ko", "tko", "knockout", "ko/tko", "stoppage"),
    "Submission": ("submission", "sub"),
    "Decision": ("decision",),
}
#: "the model", "our model", "model gives"... in the prose. Words like "role model" aside.
MODEL_TALK = re.compile(r"\b(?:the|our|a|this)\s+(?:\w+\s+)?model\b|\bmodel(?:'s)?\s+(?:says|gives|grades|graded|likes|sees|believes|has|makes|projects|favou?rs|picks|rates)\b", re.I)
#: The story citing its own data source ("per our database").
DATA_TALK = re.compile(r"\b(?:our|the) (?:database|data|stats data|records)\b|\bper our\b", re.I)
#: An outlet credit in a news story ("per Yahoo Sports", "according to MMA Fighting").
CREDIT = re.compile(r"\b(?:per|according to)\s+(?:the\s+)?[A-Z]")
#: Numbers that are not stats: event numbers, years, dates, small counts.
NOT_STATS = [
    re.compile(r"\bUFC\s+\d+"),
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}\b"),
    re.compile(r"\bRound\s+\d\b|\bR\d\b", re.I),
]


INCH = re.compile(r"(\d)[\"\u201d](?=\W|$)")


def _strip_non_prose(md: str) -> tuple[str, list[tuple[str, str]]]:
    """Article without its fenced blocks, and the blocks. Inch marks (75", 5' 11") become
    "in" so they are not read as quotation marks."""
    blocks = [(m.group(1), m.group(2)) for m in FENCE.finditer(md)]
    return INCH.sub(r"\1in", FENCE.sub("\n", md)), blocks


def _numbers(text: str) -> list[str]:
    for pat in NOT_STATS:
        text = pat.sub(" ", text)
    text = QUOTED.sub(" ", text)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", text)  # links
    return re.findall(r"(?<![\w.])\d+(?:\.\d+)?", text)


def _allowed(v: float, allowed: set[float]) -> bool:
    if v <= 12 and float(v).is_integer():
        return True  # rounds, counts, "five fights"
    return any(abs(v - a) <= 0.051 for a in allowed)


def check_code(md: str, fs: FactSheet, claims: list[Claim], news: bool = False) -> tuple[list[str], list[str]]:
    """(hard issues, unmatched numbers for the reviewer). `news`: a news-desk story, which
    has no prediction line."""
    issues, unmatched = [], []
    prose, blocks = _strip_non_prose(md)

    # 1. Prediction line (previews only)
    m = PRED.search(md)
    if news:
        pass
    elif not m:
        issues.append("The bold '**Prediction: Name by Method.**' line is missing.")
    else:
        who, how = m.group(1).lower(), m.group(2).lower()
        if fs.pick_last.lower() not in who:
            issues.append(f"The prediction line names '{m.group(1)}' but the model picks {fs.pick_name}.")
        if fs.method and not any(w in how for w in METHOD_WORDS.get(fs.method, (fs.method.lower(),))):
            issues.append(f"The prediction line says '{m.group(2)}' but the model's method is {fs.method}.")

    # 2. The model is never part of the story: not in the headline or the prose (the data
    # blocks may label their numbers "Model").
    for line in prose.splitlines():
        if MODEL_TALK.search(line) and not PRED.search(line):
            where = "headline" if line.lstrip().startswith("#") else "text"
            issues.append(f"The {where} talks about the model (\"{line.strip()[:90]}\"). Never mention the model; "
                          f"say what the numbers or the stats show instead.")
            break

    if news:
        for line in prose.splitlines():
            if DATA_TALK.search(line):
                issues.append(f"Don't cite our own data (\"{line.strip()[:80]}\"); state the fact plainly.")
                break
        credits = len(CREDIT.findall(QUOTED.sub(" ", prose)))
        if credits > 2:
            issues.append(f"The story credits outlets {credits} times (\"per ...\", \"according to ...\"). "
                          f"Widely reported facts are stated plainly; keep at most two credits, for exclusives.")

    # 3. News stories: records ("9-4", "13-4-0") and ranks ("#4", "No. 5") must appear in
    # our data as written; they are small numbers, which the general rule below lets
    # through. (Previews derive records like "5-0 in the UFC" from the fight list.)
    facts_text = fs.stats_prompt or ""
    if news:
        fact_ranks = set(re.findall(r"(?:#|No\.\s?|rank(?:ed)?\s)(\d+)", facts_text, re.I))
        bare = QUOTED.sub(" ", prose) + "\n" + "\n".join(b for lang, b in blocks if lang in ("chin-stats", "chin-takeaways"))
        for rec in set(re.findall(r"(?<![\w.-])\d{1,2}-\d{1,2}(?:-\d{1,2})?(?![\w-])", bare)):
            if not re.search(rf"(?<![\d-]){re.escape(rec)}(?:-0)?(?![\d-])", facts_text):
                issues.append(f"The record or score {rec} is not in our stats data; use the one from the stats data or drop it.")
        for r in set(re.findall(r"(?:#|No\.\s?)(\d+)\b", bare)):
            if r not in fact_ranks:
                issues.append(f"The rank #{r} is not in our stats data; use the rank from the stats data or drop it.")
    if sum(1 for lang, _ in blocks if lang == "chin-quote") > 2:
        issues.append("Use at most two chin-quote pull quotes.")

    # 3b. Numbers
    news_text = " ".join(_norm(c.source_text or "") + " " + _norm(c.text) for c in claims)
    data_blocks = "\n".join(b for lang, b in blocks if lang in ("chin-stats", "chin-bars", "chin-takeaways"))
    for raw in _numbers(prose) + _numbers(data_blocks):
        v = float(raw)
        if _allowed(v, fs.allowed):
            continue
        if re.search(rf"(?<![\w.]){re.escape(raw)}(?![\w.])", news_text):
            issues.append(f"The number {raw} is not in our stats data but appears in the news reporting; "
                          f"remove it or replace it with a number from the stats data.")
        else:
            unmatched.append(raw)

    # 4. Quotes in prose and pull quotes
    quotes = [q for q in QUOTED.findall(prose) if len(q.split()) >= 3]
    for lang, body in blocks:
        if lang == "chin-quote":
            lines = [l for l in body.strip().splitlines() if l.strip()]
            if len(lines) < 2:
                issues.append("A chin-quote block needs the quote and an attribution line.")
                continue
            q, attr = " ".join(lines[:-1]), lines[-1]
            owner = _owner(q, claims)
            if owner is None:
                issues.append(f"Pull quote not found word for word in any source: \"{q[:80]}\"")
            elif owner.speaker and owner.speaker.split()[-1].lower() not in attr.lower():
                issues.append(f"Pull quote \"{q[:50]}\" is attributed to the wrong person; "
                              f"it was said by {owner.speaker}.")
    for q in quotes:
        owner = _owner(q, claims)
        if owner is None:
            issues.append(f"Quoted words not found in any source story: \"{q[:80]}\". "
                          f"Use the exact words of a claim's QUOTE or drop the quotation marks.")
            continue
        if owner.speaker:
            para = next((p for p in prose.split("\n\n") if q in p), "")
            if owner.speaker.split()[-1] not in para:
                issues.append(f"The quote \"{q[:50]}\" must name its speaker, {owner.speaker}, in the same paragraph.")

    # 5. Embeds
    found = {c.embed_url for c in claims if c.embed_url}
    for lang, body in blocks:
        if lang in ("chin-video", "chin-tweet") and body.strip() and body.strip() not in found:
            issues.append(f"{lang} uses a link the reporting did not find: {body.strip()[:80]}")
        if lang == "chin-history" and not fs.meetings:
            issues.append("chin-history is only for fighters who have met before; these two have not.")
    return issues, unmatched


def _owner(q: str, claims: list[Claim]) -> Claim | None:
    for c in claims:
        if c.quote and (quote_in(q, c.quote) or quote_in(c.quote, q)) and c.source_text and quote_in(q, c.source_text):
            return c
    for c in claims:
        if c.source_text and quote_in(q, c.source_text):
            return c
    return None


REVIEW = """You fact-check an MMA preview before it publishes. You get the stats data, the
numbered claims [C#] the writer was given, the model's pick, and the article.

Return JSON:
  "argues_for_pick": true if the article as a whole builds the case for the model's pick and
                     method, without hedging or making the other fighter the likelier winner
  "statements": every statement of fact in the article about events, titles, injuries,
                history, records, streaks or what someone said or did (skip analysis and
                opinion grounded in the stats). For each:
                {"statement": "...", "support": "C12" (the claim that says it) or "stats" (it is in
                 the stats data) or "none" (neither says it)}
                Look through every claim before answering "none"; a claim that says the same
                thing in other words counts as support.
  "underivable_numbers": from NUMBERS TO VERIFY, the ones that cannot be found in or worked out
                 from the stats data (empty list if all are fine)"""


def review(meter: llm.Meter, md: str, fs: FactSheet, claims: list[Claim], unmatched: list[str],
           news_title: str | None = None) -> dict:
    from app.services.ufc.newsroom.writer import claims_block, thesis
    head = f"NEWS STORY (no pick; treat argues_for_pick as true): {news_title}" if news_title else f"MODEL PICK: {thesis(fs)}"
    user = f"""{head}

STATS DATA:
{fs.stats_prompt}

CLAIMS:
{claims_block(claims)}

NUMBERS TO VERIFY: {', '.join(dict.fromkeys(unmatched)) or '(none)'}

ARTICLE:
{md}"""
    return llm.chat_json(meter, "review", fs.fight_id, REVIEW, user, max_tokens=1200)


def check(meter: llm.Meter, md: str, fs: FactSheet, claims: list[Claim], news_title: str | None = None) -> CheckReport:
    issues, unmatched = check_code(md, fs, claims, news=news_title is not None)
    rv = review(meter, md, fs, claims, unmatched, news_title) if not issues else {}
    if rv:
        if rv.get("argues_for_pick") is False and news_title is None:
            issues.append(f"The article must argue for {fs.pick_name} by {fs.method}; it hedges or leans "
                          f"the other way. Make every section build that case.")
        ids = {f"C{c.id}" for c in claims}
        for st in rv.get("statements") or []:
            if not isinstance(st, dict):
                continue
            sup = str(st.get("support") or "none").strip()
            if sup == "stats" or sup in ids:
                continue
            issues.append(f"Not supported by the reporting or the stats; remove it: {st.get('statement')}")
            if len(issues) >= 8:
                break
        for n in (rv.get("underivable_numbers") or [])[:8]:
            issues.append(f"The number {n} is not in the stats data; remove it or use a number that is.")
    return CheckReport(ok=not issues, issues=issues, review=rv)
