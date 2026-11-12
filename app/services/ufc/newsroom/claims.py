"""Claims: what a news story says about the fighters, with its source.

One DeepSeek call per story. Quotes must be copied word for word: any quote that is not
found in the story's text is dropped here, before an editor ever sees it.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import unicodedata

from app.services.ufc.newsroom import llm
from app.services.ufc.newsroom.schemas import KINDS, VENUES, Article, Claim

log = logging.getLogger(__name__)

SYSTEM = """You are a research assistant on an MMA news desk. You read one news story and
pull out the context a writer previewing an upcoming fight would want: storylines, history
between the fighters, injuries, camp or coach changes, stakes, trash talk and notable quotes.

Return JSON: {"claims": [ ... ]}, at most 6 claims, most newsworthy first. Each claim:
  "about": "red", "blue" or "both" (which fighter it concerns; "both" for the matchup itself)
  "kind": one of storyline, history, injury, camp, quote, beef, report
  "text": one or two plain sentences stating the fact, in past tense, self-contained (name the
          fighters, say when and where if the story does). No fight statistics, betting odds or
          predicted percentages: those come from elsewhere.
  "quote": an exact quote from the story, copied character for character, or null. Only
           quotes that say something specific (a jab at the opponent, a plan, an injury update,
           a grievance). Never generic lines like "I'm going to win" or "I'm ready".
  "speaker": who said the quote, or null
  "show": the podcast, show or programme the quote came from if the story names it, or null
  "reporter": the interviewer or reporter the story credits (e.g. "Ariel Helwani"), or null
  "venue": one of press_conference, interview, podcast, social, report
  "newsworthiness": 1 (trivia) to 5 (the story of the fight)

Only include claims about the two named fighters. If the story has nothing useful about
them, return {"claims": []}."""


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-").replace("…", "...")
    return re.sub(r"\s+", " ", s).strip().lower()


def _bare(s: str) -> str:
    """Letters, digits and single spaces: punctuation never decides a match ("F— him." in
    a quote vs "F- him," in the story)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", _norm(s))).strip()


def quote_in(quote: str, text: str) -> bool:
    q = _bare(quote)
    return len(q) >= 8 and f" {q} " in f" {_bare(text)} "


def verbatim(quote: str, text: str) -> str | None:
    """The quote if the story has it word for word. Otherwise the longest run of its
    sentences that the story has without a break (a model often joins two halves of a
    quote across "he said"), if that run is at least five words; else None."""
    if quote_in(quote, text):
        return quote
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", quote) if p.strip()]
    best = None
    for i in range(len(parts)):
        for j in range(len(parts), i, -1):
            run = " ".join(parts[i:j])
            if len(run.split()) >= 5 and quote_in(run, text):
                if best is None or len(run) > len(best):
                    best = run
                break
    return best


def extract(meter: llm.Meter, article: Article, red: dict, blue: dict, event_name: str,
            search_tier: str, fight_id: int | None = None) -> list[Claim]:
    """`red`/`blue`: {id, name, last_fight}. Returns validated claims (not yet stored)."""
    user = f"""Upcoming fight: {red['name']} (red) vs {blue['name']} (blue), {event_name}.

Story: {article.title or ''}
Outlet: {article.outlet or ''}{f' | By {article.author}' if article.author else ''}{f' | {article.published_at}' if article.published_at else ''}

{article.text[:12000]}"""
    data = llm.chat_json(meter, "claims", fight_id, SYSTEM, user, max_tokens=2000)
    out = []
    for c in (data.get("claims") or [])[:6]:
        if not isinstance(c, dict) or not (c.get("text") or "").strip():
            continue
        about = c.get("about")
        if about not in ("red", "blue", "both"):
            continue
        quote = (c.get("quote") or "").strip().strip('"“”') or None
        if quote:
            found = verbatim(quote, article.text)
            if found is None:
                log.info(f"Dropped a quote not found verbatim in {article.url}: {quote[:60]!r}")
            quote = found
        kind = c.get("kind") if c.get("kind") in KINDS else "storyline"
        if kind == "quote" and not quote:
            continue
        subject = blue if about == "blue" else red
        other = red if subject is blue else blue
        out.append(Claim(
            fighter_id=subject["id"], opponent_id=other["id"] if about == "both" or search_tier == "matchup" else None,
            tier=_tier(about, search_tier, article.published_at, subject.get("last_fight")),
            kind=kind, text=c["text"].strip(), quote=quote,
            speaker=(c.get("speaker") or None) if quote else None,
            outlet=article.outlet, show=c.get("show") or None, reporter=c.get("reporter") or None,
            venue=c.get("venue") if c.get("venue") in VENUES else "report",
            source_url=article.url, published_at=article.published_at,
            newsworthiness=max(1, min(5, int(c.get("newsworthiness") or 3))),
            source_text=article.text,
        ))
        # A "both" claim belongs to the matchup; record it against blue too so blue's
        # ledger carries it.
        if about == "both":
            out.append(out[-1].model_copy(update={"fighter_id": other["id"], "opponent_id": subject["id"]}))
    return out


def _tier(about: str, search_tier: str, published: dt.date | None, last_fight: dt.date | None) -> str:
    if about == "both" or search_tier == "matchup":
        return "matchup"
    if published and last_fight and published <= last_fight:
        return "background"
    return "recent" if search_tier != "background" else "background"
