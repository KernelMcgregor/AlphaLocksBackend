"""One DeepSeek call per run: what each candidate story is, and which are worth writing.

The model sees only headlines (no pages), names each story, sorts it into a kind, folds
spin-offs into their parent story (a pundit's take on a brawl belongs with the brawl),
throws out fight-week coverage the preview articles already cover, and suggests one
follow-up search, which is kept only if it names the story's fighters.
"""

from __future__ import annotations

import logging

from app.services.ufc.newsroom import llm
from app.services.ufc.newsroom.desk.topics import Cluster

log = logging.getLogger(__name__)

KINDS = ("incident", "trash_talk", "press_conference", "rumored_fight", "booking", "callout", "injury",
         "contract", "controversy", "personal", "preview", "other")
#: Never written: previews and analysis are the preview articles' job; lifestyle is not news.
WRITE_KINDS = {"incident", "trash_talk", "press_conference", "rumored_fight", "booking", "callout", "injury",
               "contract", "controversy", "other"}

SYSTEM = """You are the news editor of a UFC site. You get candidate stories, each a cluster of
headlines from the last day or two. For each one decide what it is.

Return JSON: {"stories": [ ... ]}, one entry per candidate, each:
  "id": the candidate's number
  "title": a short neutral working title (who, what)
  "kind": one of incident, trash_talk, press_conference, rumored_fight, booking, callout,
          injury, contract, controversy, personal, preview, other
          trash_talk = fighters going at each other: insults, threats, promises, beef
          press_conference = notable things said at a press conference, media day or in an
          interview (a vow, a revelation, a retraction, a jab)
          preview = predictions, picks, odds, matchup breakdowns, weigh-in results, results
          recaps, how to watch (another desk covers those)
          personal = lifestyle, net worth, family, fan interactions
  "newsworthy": true only for real news a fight fan would click: something happened or
          was said that changes or reveals something. What fighters SAY (press conferences,
          media days, trash talk, callouts, statements) is newsworthy even in fight week;
          analysis and predictions are not.
  "merge_into": the id of another candidate this is a spin-off of (reactions, pundit takes
          or knock-on effects of the same event), else null
  "follow_up": one Google News search that would find more on this story, using the
          fighters' full names and one or two key words, else null"""


def label(meter: llm.Meter, clusters: list[Cluster], names: dict[int, str], limit: int = 15) -> list[dict]:
    cands = clusters[:limit]
    blocks = []
    for i, c in enumerate(cands):
        who = ", ".join(names.get(f, "?") for f in c.fighter_ids[:4]) or "none of our fighters"
        heads = "\n".join(f"   - {h}" for h in c.headlines[:6])
        blocks.append(f"[{i}] fighters: {who} | {len(c.outlets)} outlets\n{heads}")
    data = llm.chat_json(meter, "desk_label", None, SYSTEM, "\n\n".join(blocks), max_tokens=2500)
    out = []
    for s in data.get("stories") or []:
        try:
            i = int(s.get("id"))
        except (TypeError, ValueError):
            continue
        if not 0 <= i < len(cands):
            continue
        merge = s.get("merge_into")
        try:
            merge = int(merge) if merge is not None else None
        except (TypeError, ValueError):
            merge = None
        if merge == i or (merge is not None and not 0 <= merge < len(cands)):
            merge = None
        follow = (s.get("follow_up") or "").strip() or None
        fighters = [names[f] for f in cands[i].fighter_ids if f in names]
        if follow and fighters and not any(n.split()[-1].lower() in follow.lower() for n in fighters):
            follow = None  # must stay on the story's own fighters
        out.append({"id": i, "title": (s.get("title") or "").strip() or cands[i].headlines[0],
                    "kind": s.get("kind") if s.get("kind") in KINDS else "other",
                    "newsworthy": bool(s.get("newsworthy")), "merge_into": merge, "follow_up": follow})
    return out


def pick(clusters: list[Cluster], labels: list[dict], top: int = 3) -> list[tuple[Cluster, dict]]:
    """Fold spin-offs into their parents, then the best `top` newsworthy stories by score."""
    by_id = {lab["id"]: lab for lab in labels}
    writable = {i for i, lab in by_id.items() if lab["newsworthy"] and lab["kind"] in WRITE_KINDS}
    # A spin-off only folds into a parent that will be written; one pointing at skipped
    # fight-week coverage stands on its own instead of vanishing with it.
    for lab in labels:
        if lab["merge_into"] is not None and lab["merge_into"] not in writable:
            lab["merge_into"] = None
    merged = {i: [c] for i, c in enumerate(clusters)}
    for lab in labels:
        p = lab["merge_into"]
        # Follow chains (a -> b -> c) to the root parent.
        seen = set()
        while p is not None and p in by_id and by_id[p]["merge_into"] is not None and p not in seen:
            seen.add(p)
            p = by_id[p]["merge_into"]
        if p is not None and p in merged and lab["id"] in merged and p != lab["id"]:
            merged[p].extend(merged.pop(lab["id"]))
    out = []
    for i, group in merged.items():
        lab = by_id.get(i)
        if not lab or not lab["newsworthy"] or lab["kind"] not in WRITE_KINDS:
            continue
        head = group[0]
        if len(group) > 1:
            items = [it for g in group for it in g.items]
            fids = list(dict.fromkeys(f for g in group for f in g.fighter_ids))
            head = Cluster(items=items, fighter_ids=fids, outlets=sorted({o for g in group for o in g.outlets}),
                           newest=max((g.newest for g in group if g.newest), default=None),
                           score=sum(g.score for g in group), key=head.key)
        out.append((head, lab))
    return sorted(out, key=lambda x: -x[0].score)[:top]
