"""The editor plans the article; the writer drafts and, when the checker objects, revises.

The writer keeps every rule of the single-call previews (build_preview_prompt's system
prompt: sections, chin blocks, no model internals, the bold prediction line) and adds the
newsroom's: the story frames, the stats argue, every quote attributed, no number that is
not in the stats block.
"""

from __future__ import annotations

import json
import re

from app.services.ufc.newsroom import llm
from app.services.ufc.newsroom.facts import FactSheet
from app.services.ufc.newsroom.schemas import Claim, Plan


def claims_block(claims: list[Claim]) -> str:
    lines = []
    for c in claims:
        src = ", ".join(x for x in (c.outlet, c.show, f"reporter {c.reporter}" if c.reporter else None,
                                    str(c.published_at) if c.published_at else None) if x)
        q = f' | QUOTE ({c.speaker}): "{c.quote}"' if c.quote else ""
        e = f" | EMBED {c.embed_url}" if c.embed_url else ""
        lines.append(f"[C{c.id}] ({c.tier}, {c.kind}, newsworthiness {c.newsworthiness}) {c.text}{q} | SOURCE: {src}{e}")
    return "\n".join(lines) or "(none)"


def thesis(fs: FactSheet) -> str:
    edge = f", market {fs.market_prob * 100:.1f}% ({fs.edge_pts:+.1f} pts)" if fs.market_prob is not None else ""
    return f"{fs.pick_name} by {fs.method} (model {fs.pick_prob * 100:.1f}%{edge})"


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------

EDITOR = """You are the editor of a data-driven MMA site. Our niche is the numbers: every
article argues for our model's pick and method. Reporting gives it a story; it never
changes the verdict.

From the sourced claims, plan the article and return JSON:
  "angle": one sentence, the story of this fight that the article leads with
  "headline_hook": a specific, original idea for the headline's one-liner
  "lead_claims": claim ids (numbers) the article is built around: matchup and recent tier first
  "background_claims": at most 2 background-tier ids, only if a reader needs them to understand
                        this fight (e.g. how the champion won the belt)
  "pull_quotes": at most 2 claim ids with a quote worth showing as a highlighted pull quote:
                 trash talk, a vow, a grievance. Never a generic boast.
  "counterpoints": claim ids that point against our pick (a pundit picking the other fighter,
                   an opponent's warning); the article answers each with a stat
  "embeds": up to 2 EMBED urls from the claims worth embedding (press conference clips, posts)

Drop claims that are trivial, duplicated or only loosely about this fight."""


def plan(meter: llm.Meter, fs: FactSheet, claims: list[Claim]) -> Plan:
    meetings = "\n".join(f"- {m['event']} ({m['date']}): {m['winner']} by {m['method']}" for m in fs.meetings)
    user = f"""Fight: {fs.red['name']} vs {fs.blue['name']}, {fs.event_name}
Our pick: {thesis(fs)}
{f'Previous meetings:{chr(10)}{meetings}' if meetings else ''}

Claims:
{claims_block(claims)}"""
    data = llm.chat_json(meter, "editor", fs.fight_id, EDITOR, user, max_tokens=1200)
    ids = {c.id for c in claims}
    embeds = {c.embed_url for c in claims if c.embed_url}

    def keep(key, n):
        out = []
        for v in data.get(key) or []:
            try:
                v = int(str(v).lstrip("C"))
            except ValueError:
                continue
            if v in ids and v not in out:
                out.append(v)
        return out[:n]

    return Plan(angle=data.get("angle") or "", headline_hook=data.get("headline_hook") or "",
                lead_claims=keep("lead_claims", 8), background_claims=keep("background_claims", 2),
                pull_quotes=keep("pull_quotes", 2), counterpoints=keep("counterpoints", 3),
                embeds=[u for u in (data.get("embeds") or []) if u in embeds][:2])


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

NEWSROOM_RULES = """

NEWSROOM RULES (these add to everything above):
- You also get REPORTING: sourced claims about the fighters and the editor's plan. Lead with
  the editor's angle. Weave the reporting into the Overview and the fighter sections so the
  reader knows the story: the history between them, how a title changed hands, injuries,
  what each has said. Background claims get one or two sentences at most.
- The reporting frames the fight; the stats decide it. Every section still builds the case
  for the MODEL PREDICTION and METHOD PREDICTION. A counterpoint (a pundit picking the other
  fighter, a warning from the opponent) is stated fairly, then answered with a number.
- Every number in the article comes from the stats data above (records, model and market
  probabilities, rating percentiles, fight stats, round and time of earlier finishes). Never
  take a number from the reporting, except inside a verbatim quote.
- Attribute like a reporter: name the speaker and where they said it ("Kape told MMA Junkie",
  "Pantoja said on The Ariel Helwani Show", "per ESPN's Marc Raimondi"). Quote only the
  exact words given in a claim's QUOTE, never paraphrase inside quotation marks, never invent
  a quote. Do not report claims that are not in the REPORTING.
- New blocks you may use (in addition to the three above):

```chin-quote
Exact words of the quote
Speaker | Outlet or show | Month YYYY | source url
```
(A highlighted pull quote. Only for the editor's pull_quotes. At most two per article, never
two in a row, placed inside the Overview or a fighter section next to the paragraph it belongs
to, never after "How This Fight Plays Out". Include the source url when the claim has one.)

```chin-history
```
(Leave empty. Shows the earlier meetings between these two fighters from our database. Use
once, near the top, only when PREVIOUS MEETINGS are listed.)

```chin-video
https://www.youtube.com/watch?v=...
```
```chin-tweet
https://x.com/user/status/...
```
(Only for the editor's embeds, exactly as given.)

```chin-movement
```
(Leave empty. The live market line chart. Use once, in the section that compares the model
with the market.)
"""


def write(meter: llm.Meter, fs: FactSheet, claims: list[Claim], p: Plan) -> str:
    by_id = {c.id: c for c in claims}
    used = [by_id[i] for i in dict.fromkeys(p.lead_claims + p.background_claims + p.pull_quotes + p.counterpoints)]
    user = fs.stats_prompt + f"""

**REPORTING (sourced claims):**
{claims_block(used)}

**EDITOR'S PLAN:**
{json.dumps(p.model_dump(), indent=2)}"""
    out = llm.chat(meter, "writer", fs.fight_id, fs.system_prompt + NEWSROOM_RULES, user, max_tokens=5000)
    return _unfence(out)


def revise(meter: llm.Meter, fs: FactSheet, claims: list[Claim], p: Plan, draft: str, issues: list[str]) -> str:
    by_id = {c.id: c for c in claims}
    used = [by_id[i] for i in dict.fromkeys(p.lead_claims + p.background_claims + p.pull_quotes + p.counterpoints)]
    user = fs.stats_prompt + f"""

**REPORTING (sourced claims):**
{claims_block(used)}

**DRAFT:**
{draft}

**THE FACT-CHECKER REJECTED THE DRAFT FOR THESE REASONS:**
""" + "\n".join(f"- {i}" for i in issues) + """

Fix every problem listed (remove or correct the offending number, quote or sentence) and change
nothing else. Return the full corrected article in markdown, nothing else."""
    out = llm.chat(meter, "revise", fs.fight_id, fs.system_prompt + NEWSROOM_RULES, user, max_tokens=5000)
    return _unfence(out)


def _unfence(s: str) -> str:
    return re.sub(r"^```(?:markdown|md)?\s*\n([\s\S]*?)\n```$", r"\1", s.strip())
