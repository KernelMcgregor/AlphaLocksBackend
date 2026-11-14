"""Research and write one news story: what fighters did and said, not fight analysis.

    research   the follow-up search plus the cluster's own coverage, the best pages read,
               claims extracted (verbatim quotes only)
    facts      for each fighter: record, champion or division rank, last results, next
               booked fight; the only numbers the story may use
    write      a news report; checked by the preview newsroom's checker in news mode
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.services.ufc.newsroom import checker, claims as claims_mod, llm, wire
from app.services.ufc.newsroom.desk.topics import BIG, Cluster
from app.services.ufc.newsroom.facts import FactSheet, _expand, _ratios, numbers_in
from app.services.ufc.newsroom.schemas import Claim, NewsItem
from app.services.ufc.newsroom.writer import claims_block

log = logging.getLogger(__name__)

MAX_READ = 8


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------

def _rank_items(items: list[NewsItem]) -> list[NewsItem]:
    seen, out = set(), []
    for it in sorted(items, key=lambda i: ((i.outlet or "").lower() not in BIG,
                                           -(i.published_at or dt.date.min).toordinal())):
        k = re.sub(r"\W+", " ", it.title.lower())[:70]
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def research(meter: llm.Meter, cluster: Cluster, label: dict, fighters: list[dict], match) -> tuple[list, list[Claim]]:
    items = list(cluster.items)
    if label.get("follow_up"):
        items += wire.google_news(f"{label['follow_up']} when:3d")
    picked = _rank_items(items)[: MAX_READ * 2]

    def read(it: NewsItem):
        url = wire.resolve(it.link)
        art = wire.fetch(url, it) if url else None
        return it, art

    with ThreadPoolExecutor(max_workers=6) as pool:
        got = [(it, a) for it, a in pool.map(read, picked) if a is not None][:MAX_READ]

    def extract(pair):
        it, art = pair
        try:
            return claims_mod.extract_topic(meter, art, fighters, label["title"], match)
        except llm.BudgetExceeded:
            raise
        except Exception as e:
            log.warning(f"Extraction failed for {art.url}: {e}")
            return []

    with ThreadPoolExecutor(max_workers=6) as pool:
        cs = [c for batch in pool.map(extract, got) for c in batch]
    # One claim per fact: syndicated copies of the same story say the same thing.
    seen, out = set(), []
    for c in sorted(cs, key=lambda c: -c.newsworthiness):
        k = (c.quote or c.text)[:60].lower()
        if k not in seen:
            seen.add(k)
            out.append(c)
    for i, c in enumerate(out, 1):
        c.id = i
    return [a for _, a in got], out


def central(fighters: list[dict], cs: list[Claim]) -> list[dict]:
    """The fighters the story is about: the lead fighter plus anyone who is the subject of
    two or more claims. A fighter named once in a spin-off gets no fact-sheet line."""
    count: dict[int, int] = {}
    for c in cs:
        count[c.fighter_id] = count.get(c.fighter_id, 0) + 1
    return [f for i, f in enumerate(fighters) if i == 0 or count.get(f["id"], 0) >= 2]


# ---------------------------------------------------------------------------
# Facts: the fighters' numbers, from our tables only
# ---------------------------------------------------------------------------

def news_facts(db: Session, fighter_ids: list[int]) -> FactSheet:
    from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFighterRanking
    from app.services.ufc.champions import current_champions

    champs = {fid: div for div, fid in current_champions(db).items()}
    today = dt.date.today()
    lines = []
    for fid in fighter_ids:
        f = db.get(UFCFighter, fid)
        if f is None:
            continue
        rank = db.query(UFCFighterRanking).filter(UFCFighterRanking.fighter_id == fid).order_by(UFCFighterRanking.rank).first()
        standing = (f"current UFC {champs[fid].replace('_', ' ')} champion" if fid in champs
                    else f"#{rank.rank} in the {rank.weight_class.replace('_', ' ').replace('w ', 'women s ')} rankings" if rank else "unranked")
        fights = (db.query(UFCFight, UFCEvent).join(UFCEvent, UFCEvent.id == UFCFight.event_id)
                  .filter(or_(UFCFight.red_fighter_id == fid, UFCFight.blue_fighter_id == fid))
                  .order_by(UFCEvent.date.desc()).limit(6).all())
        past, upcoming = [], None
        for fight, ev in fights:
            opp = fight.blue_fighter if fight.red_fighter_id == fid else fight.red_fighter
            oname = f"{opp.first_name} {opp.last_name}" if opp else "TBA"
            if fight.method is None and ev.date >= today:
                upcoming = f"{ev.name} on {ev.date} vs {oname}"
            elif fight.method:
                res = "won" if fight.winner_id == fid else "lost" if fight.winner_id else "no decision"
                past.append(f"{ev.name} ({ev.date}): {res} vs {oname} by {fight.method}, R{fight.finish_round} {fight.finish_time}")
        lines.append(f"- {f.first_name} {f.last_name}: record {f.wins}-{f.losses}-{f.draws}; {standing}; "
                     f"last results: {'; '.join(past[:3]) or 'none in the UFC'}; next fight: {upcoming or 'none booked'}")
    text = "**FIGHTERS (the only source for numbers; never name this list in the story):**\n" + "\n".join(lines)
    allowed = _expand(numbers_in(text)) | _ratios(text)
    for m, s in re.findall(r"\b(\d+):(\d{2})\b", text):
        allowed |= {int(m) * 60 + int(s), int(s), int(m)}
    stub = {"id": 0, "name": "", "last": ""}
    return FactSheet(fight_id=0, context={}, stats_prompt=text, system_prompt="", red=stub, blue=stub,
                     event_name="", card_position=None, is_title=False, pick_corner=None, pick_name=None,
                     pick_last=None, pick_prob=None, method=None, market_prob=None, allowed=allowed)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

WRITER = """You are a reporter on a UFC news site. Write a news story from the REPORTING you are
given: what happened and what was said. This is news, not fight analysis: no predictions,
no picks, no breaking down who wins a fight.

Format (markdown):
# A specific, punchy headline (no clickbait questions, no "the model")
A one or two sentence opening paragraph that tells the reader the news.

Then 250-550 words. Use ## subheads only if the story has distinct parts (for example
"What happened", "The fallout", "What's next"). Lead with the newest development, then the
background.

Rules:
- Report only what the REPORTING says. Never invent events, quotes, dates or reactions.
- Write only about this STORY. The reporting may include other news from the same
  articles (another fighter's incident, other bouts on the card); leave it out entirely.
  No "separate incident" or roundup sections.
- Credit an outlet only for its own exclusive: a one-on-one interview ("told Ariel Helwani
  on The Ariel Helwani Show"), a scoop or a report from sources ("per ESPN's Marc
  Raimondi"). Rumors name who reported them. Everything else is widely reported: state it
  plainly, and attribute quotes to the speaker and where they said it ("Hokit said at the
  press conference", "Dumont wrote on Instagram"), never to the outlet that wrote it up.
  At most two outlet credits in the whole story.
- Quote only the exact words of a claim's QUOTE. At most two pull quotes, as:
```chin-quote
Exact words of the quote
Speaker | Where it was said (press conference, Instagram, or the outlet for an exclusive interview) | Month YYYY | source url
```
- Embed a tweet or video only with an EMBED url given in the reporting:
```chin-tweet
https://x.com/user/status/...
```
```chin-video
https://www.youtube.com/watch?v=...
```
- Numbers (records, ranks, fight results, rounds, times) only from FIGHTERS below; never a
  number from the reporting except inside a verbatim quote. Use them only where they help
  the reader place a fighter ("the former bantamweight champion, 20-4").
- State numbers as plain facts ("Dumont, 13-4, fifth in the women's bantamweight rankings").
  Never mention a database, "our data", "the stats" or where the numbers came from.
- A fighter's numbers belong to that fighter only: name them in full when you use them.
  People not listed under FIGHTERS (fighters from other promotions, officials, coaches)
  have no numbers; never give them a record or a rank.
- Mind the dates: something scheduled after today (TODAY below) is upcoming ("will face",
  "is set to"), not past.
- Never mention a model, predictions, odds or betting.
- Neutral, factual tone on legal and personal matters (arrests, sentences, health): say what
  was reported and by whom, nothing more. No moralising, no speculation.
- No em dashes. No captions, footnotes or source lists."""


def write(meter: llm.Meter, facts: FactSheet, cs: list[Claim], label: dict) -> str:
    use = sorted(cs, key=lambda c: -c.newsworthiness)[:18]
    user = f"""TODAY: {dt.date.today():%A, %B %d, %Y}
STORY: {label['title']} ({label['kind'].replace('_', ' ')})

{facts.stats_prompt}

REPORTING (sourced claims):
{claims_block(use)}"""
    out = llm.chat(meter, "desk_write", None, WRITER, user, max_tokens=3000)
    return re.sub(r"^```(?:markdown|md)?\s*\n([\s\S]*?)\n```$", r"\1", out.strip())


def revise(meter: llm.Meter, facts: FactSheet, cs: list[Claim], label: dict, draft: str, issues: list[str]) -> str:
    use = sorted(cs, key=lambda c: -c.newsworthiness)[:18]
    user = f"""TODAY: {dt.date.today():%A, %B %d, %Y}
STORY: {label['title']}

{facts.stats_prompt}

REPORTING (sourced claims):
{claims_block(use)}

DRAFT:
{draft}

THE FACT-CHECKER REJECTED THE DRAFT FOR THESE REASONS:
""" + "\n".join(f"- {i}" for i in issues) + """

Fix every problem listed and change nothing else. Return the full corrected story in markdown."""
    out = llm.chat(meter, "desk_revise", None, WRITER, user, max_tokens=3000)
    return re.sub(r"^```(?:markdown|md)?\s*\n([\s\S]*?)\n```$", r"\1", out.strip())


def write_checked(meter: llm.Meter, facts: FactSheet, cs: list[Claim], label: dict, revisions: int = 2):
    draft = write(meter, facts, cs, label)
    report = checker.check(meter, draft, facts, cs, news_title=label["title"])
    first = list(report.issues)
    for _ in range(revisions):
        if report.ok:
            break
        draft = revise(meter, facts, cs, label, draft, report.issues)
        report = checker.check(meter, draft, facts, cs, news_title=label["title"])
    return draft, report, first


# ---------------------------------------------------------------------------
# Focus and format
# ---------------------------------------------------------------------------

FOCUS = """You are the news editor of a UFC site. A story has been researched. Decide whose story
it is: the fighter or fighters it is ABOUT, as opposed to people it merely mentions
(someone who reacted, an opponent named in passing, a pundit's subject).

Examples: "Norma Dumont leaves the UFC; Amanda Nunes calls it a bad decision" is about
Dumont only. "Tom Aspinall predicts Gane vs Hokit" is about Aspinall only. "Usman
Nurmagomedov attacks Merab Dvalishvili" is about both. A rumored or booked fight is about
both fighters in it. A story about the relationship between people is about all of them:
"Borralho says his friendship with Pereira survived Pereira cornering his opponent
Belgaroui" is about Borralho, Pereira and Belgaroui.

Then list the claims that are not part of this story at all: other news that happened to
appear in the same articles (another fighter's incident, other bouts on the card, unrelated
history). Background that explains this story stays.

Return JSON: {"focus": [full names, the main subject first, at most 4],
              "unrelated": [claim ids like "C7"]}"""

PAIR_KINDS = {"rumored_fight", "booking", "callout", "trash_talk"}


def choose_focus(meter: llm.Meter, label: dict, fighters: list[dict], cs: list[Claim]) -> tuple[list[dict], list[Claim]]:
    """The fighters the story is about, main subject first (a subset of `fighters`), and
    the claims that belong to the story (unrelated news from the same articles dropped)."""
    notes = "\n".join(f"[C{c.id}] {c.text}" for c in cs)
    user = f"""Story: {label['title']} ({label['kind'].replace('_', ' ')})
Fighters in our database involved: {', '.join(f['name'] for f in fighters)}

What the reporting says:
{notes}"""
    data = llm.chat_json(meter, "desk_focus", None, FOCUS, user, max_tokens=600)
    from app.services.ufc.newsroom.claims import same_person
    by_id = {f["id"]: f for f in fighters}
    out = []
    for n in data.get("focus") or []:
        f = by_id.get(same_person(str(n), fighters))
        if f and f not in out:
            out.append(f)
    drop = {str(x).strip().upper().lstrip("C") for x in data.get("unrelated") or []}
    kept = [c for c in cs if str(c.id) not in drop]
    return (out[:4] or fighters[:1]), (kept if len(kept) >= 3 else cs)


def layout_for(kind: str, focus: list[dict]) -> str:
    if len(focus) <= 1:
        return "profile"
    if len(focus) == 2:
        return "matchup" if kind in PAIR_KINDS else "pair"
    return "group"
