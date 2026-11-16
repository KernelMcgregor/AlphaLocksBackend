"""One DeepSeek call per run: what each candidate story is, and which are worth writing.

The model sees only headlines (no pages). It splits clusters into developments (each one
thing that happened or was said: its own article), names and sorts each, folds a
development reported in two clusters into one, recognises developments the site is
already running (more coverage, no new article), throws out fight-week coverage the
preview articles already cover, and suggests one follow-up search, which is kept only if
it names the story's fighters.
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
headlines from the last day or two, and the stories the site is already running.

Every DEVELOPMENT is its own article. A development is one thing that happened or was
said: an attack, a sentencing, a fighter's statement about it, a new revelation, another
fighter's reaction that is news in itself. "Usman attacks Merab", "Usman jailed", "Merab
calls for Usman's release", "Merab says Umar hurt him" and "Topuria wanted to go to Abu
Dhabi" are five developments and five articles, even though the same people are in them.
Only the same development reported by different outlets, or with small new details, is
one article.

A candidate's headlines may mix developments: split them. Return JSON
{"stories": [ ... ]}, one entry per development, each:
  "candidate": the candidate's number
  "headlines": the numbers of that candidate's headlines that report this development
          (every headline when the candidate is a single development)
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
  "running": the R number of a RUNNING story that already covers this same development
          (more coverage of it, nothing new happened), else null. A new development about
          a running story's people is NOT the running story.
  "merge_into": the number of another candidate reporting this same development, else null
  "follow_up": one Google News search that would find more on this development, using the
          fighters' full names and one or two key words, else null"""

SHOW = 10


def _part(c: Cluster, items: list) -> Cluster:
    """A candidate cut down to one development's headlines."""
    outlets = sorted({(i.outlet or "").strip() for i in items if i.outlet})
    share = len(outlets) / max(1, len(c.outlets))
    return Cluster(items=items, fighter_ids=c.fighter_ids, outlets=outlets,
                   newest=max((i.published_at for i in items if i.published_at), default=None),
                   score=round(c.score * share, 3), key=c.key)


def label(meter: llm.Meter, clusters: list[Cluster], names: dict[int, str], running: list[tuple[int, str]] = (),
          limit: int = 15) -> tuple[list[Cluster], list[dict]]:
    """Split the candidates into developments and label each: (developments, labels), a
    label's "id" indexing the developments. `running`: (topic id, title) of the stories
    the site is running."""
    cands = clusters[:limit]
    blocks = []
    for i, c in enumerate(cands):
        who = ", ".join(names.get(f, "?") for f in c.fighter_ids[:4]) or "none of our fighters"
        heads = "\n".join(f"   {j}. {h}" for j, h in enumerate(c.headlines[:SHOW]))
        blocks.append(f"[{i}] fighters: {who} | {len(c.outlets)} outlets\n{heads}")
    run_ids = {tid for tid, _ in running}
    user = ("RUNNING STORIES:\n" + ("\n".join(f"R{tid}: {t}" for tid, t in running) or "(none)")
            + "\n\nCANDIDATES:\n" + "\n\n".join(blocks))
    data = llm.chat_json(meter, "desk_label", None, SYSTEM, user, max_tokens=3500)

    def num(v):
        try:
            return int(str(v).strip().lstrip("Rr"))
        except (TypeError, ValueError):
            return None

    per: dict[int, list[dict]] = {}
    for s in data.get("stories") or []:
        i = num(s.get("candidate", s.get("id")))
        if i is not None and 0 <= i < len(cands):
            per.setdefault(i, []).append(s)
    devs, labs, first = [], [], {}
    for i, entries in sorted(per.items()):
        c = cands[i]
        shown = len(c.items[:SHOW])
        taken: set[int] = set()
        parts = []
        for s in entries:
            hs = [h for h in (num(x) for x in s.get("headlines") or []) if h is not None and 0 <= h < shown and h not in taken]
            taken.update(hs)
            parts.append((s, hs))
        if len(parts) == 1:
            parts = [(parts[0][0], list(range(len(c.items))))]
        else:
            # Headlines not shown or not placed go with the candidate's first development.
            rest = [j for j in range(len(c.items)) if j not in taken]
            parts = [(s, hs + (rest if k == 0 else [])) for k, (s, hs) in enumerate(parts)]
        for s, hs in parts:
            if not hs:
                continue
            first.setdefault(i, len(devs))
            devs.append(_part(c, [c.items[j] for j in sorted(hs)]) if len(parts) > 1 else c)
            follow = (s.get("follow_up") or "").strip() or None
            fighters = [names[f] for f in c.fighter_ids if f in names]
            if follow and fighters and not any(n.split()[-1].lower() in follow.lower() for n in fighters):
                follow = None  # must stay on the story's own fighters
            run = num(s.get("running"))
            labs.append({"id": len(devs) - 1, "candidate": i, "title": (s.get("title") or "").strip() or devs[-1].headlines[0],
                         "kind": s.get("kind") if s.get("kind") in KINDS else "other",
                         "newsworthy": bool(s.get("newsworthy")), "merge_into": num(s.get("merge_into")),
                         "running": run if run in run_ids else None, "follow_up": follow})
    # merge_into names a candidate; point it at that candidate's first development.
    for lab in labs:
        m = lab["merge_into"]
        lab["merge_into"] = first.get(m) if m is not None and m != lab["candidate"] else None
        if lab["merge_into"] == lab["id"]:
            lab["merge_into"] = None
    return devs, labs


def pick(clusters: list[Cluster], labels: list[dict], top: int = 3) -> list[tuple[Cluster, dict]]:
    """Fold duplicates into the development they repeat, then the best `top` newsworthy
    developments by score."""
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
