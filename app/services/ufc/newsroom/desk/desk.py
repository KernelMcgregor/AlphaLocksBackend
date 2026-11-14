"""Run the news desk: sweep, cluster, label, then research and write the top stories.

    python -m app.services.ufc.newsroom.desk                               # live: stores and publishes
    python -m app.services.ufc.newsroom.desk --dry-run --out /tmp/desk     # writes nothing to the DB
    python -m app.services.ufc.newsroom.desk --dry-run --discover-only     # just the story list

A story publishes only when the checker passes and at least two different outlets
reported it; otherwise it waits in the admin queue. A story already running is updated,
not duplicated: new coverage attaches to it, and it is rewritten only when enough new
outlets have joined.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.database import SessionLocal
from app.services.ufc.newsroom import llm
from app.services.ufc.newsroom.desk import entities, labeler, report, sources, store, topics

log = logging.getLogger(__name__)
MIN_OUTLETS = 2
MAX_DOSSIERS = 6


def discover(db, meter: llm.Meter, top: int = 3, persist: bool = False):
    from app.models.ufc import UFCFighterRanking
    from app.services.ufc.champions import current_champions

    matcher = entities.build(db)
    follow = store.follow_names(db, matcher.names) if persist else []
    items = topics.clean(sources.harvest(db, follow_names=follow))
    fighters = [matcher.match(i.title) for i in items]
    if persist:
        counts: dict[int, int] = {}
        for fs in fighters:
            for f in fs:
                counts[f] = counts.get(f, 0) + 1
        store.record_mentions(db, counts)
    important = set(current_champions(db).values()) | {
        f for (f,) in db.query(UFCFighterRanking.fighter_id).filter(UFCFighterRanking.rank <= 5)}
    clusters = topics.cluster(items, fighters, important)
    labels = labeler.label(meter, clusters, matcher.names)
    # A few spare picks: a story already running with little new coverage is skipped.
    return matcher, clusters, labels, labeler.pick(clusters, labels, top + 3)


def write_one(db, meter, matcher, cluster, lab, persist: bool, topic=None) -> dict:
    fighters = [{"id": f, "name": matcher.names[f]} for f in cluster.fighter_ids[:5] if f in matcher.names]
    arts, cs = report.research(meter, cluster, lab, fighters, matcher.match)
    fighters = report.central(fighters, cs)
    # Whose story it is: only they get a fact-sheet line and a profile; the rest are
    # mentioned. The count and the kind of story set the page's format.
    focus, cs = report.choose_focus(meter, lab, fighters, cs)
    layout = report.layout_for(lab["kind"], focus)
    facts = report.news_facts(db, [f["id"] for f in focus])
    draft, rep, first = report.write_checked(meter, facts, cs, lab)
    outlets = len(set(cluster.outlets))
    status = "published" if rep.ok and outlets >= MIN_OUTLETS else "needs_review"
    story = {"title": lab["title"], "kind": lab["kind"], "status": status, "layout": layout,
             "focus": [f["name"] for f in focus],
             "fighters": [f["name"] for f in fighters], "outlets": outlets, "read": len(arts),
             "claims": len(cs), "issues": rep.issues, "first_draft_issues": first, "draft": draft}
    if persist:
        row = store.save_story(db, topic, cluster, lab, fighters, arts, cs, draft,
                               {"label": lab, "issues": rep.issues, "first_draft_issues": first,
                                "review": rep.review, "read": len(arts), "claims": len(cs)}, status,
                               focus=focus, layout=layout)
        story["slug"], story["topic_id"] = row.slug, row.id
        story["cover"] = store.cover(db, row, arts)
        story["fighter_ids"] = [f["id"] for f in focus]
    return story


def run(persist: bool = False, top: int = 3, out_dir: Path | None = None, discover_only: bool = False) -> dict:
    db = SessionLocal()
    try:
        meter = llm.Meter(persist=persist, month_spent=llm.month_spent(db) if persist else 0.0)
        matcher, clusters, labels, picks = discover(db, meter, top, persist)
        summary = {"candidates": [{**lab, "headlines": clusters[lab["id"]].headlines[:5],
                                   "outlets": len(clusters[lab["id"]].outlets)} for lab in labels],
                   "stories": [], "updated": []}
        written = 0
        if not discover_only:
            for cluster, lab in picks:
                if written >= top:
                    break
                topic = store.match_topic(db, cluster) if persist else None
                if topic is not None:
                    if topic.status == "rejected" or store.new_outlets(db, topic, cluster) < store.REWRITE_NEW_OUTLETS:
                        store.attach_items(db, topic, cluster)
                        db.commit()
                        summary["updated"].append({"slug": topic.slug, "title": topic.title})
                        continue
                try:
                    story = write_one(db, meter, matcher, cluster, lab, persist, topic)
                except llm.BudgetExceeded as e:
                    log.warning(str(e))
                    break
                summary["stories"].append(story)
                written += 1
                log.info(f"{story['status']}: {lab['title']} ({story['read']} read, {story['claims']} claims)")
            if persist:
                done = set()
                for s in summary["stories"]:
                    for fid in s.get("fighter_ids", []):
                        if fid in done or len(done) >= MAX_DOSSIERS:
                            continue
                        done.add(fid)
                        try:
                            store.refresh_dossier(db, meter, fid, matcher.names.get(fid, ""))
                        except llm.BudgetExceeded:
                            break
                if summary["stories"]:
                    from app.services import response_cache
                    response_cache.invalidate()
        summary["usd"] = round(meter.run_usd, 4)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "desk.json").write_text(json.dumps(summary, indent=2, default=str))
            for i, s in enumerate(summary["stories"], 1):
                (out_dir / f"story_{i}.md").write_text(s["draft"])
        return summary
    finally:
        db.close()


def rewrite(topic_id: int) -> dict:
    """Admin: research and rewrite one stored story from its own headlines."""
    from app.models.ufc import UFCNewsTopic, UFCNewsTopicItem
    from app.services.ufc.newsroom.schemas import NewsItem

    db = SessionLocal()
    try:
        t = db.get(UFCNewsTopic, topic_id)
        if t is None:
            raise LookupError("No such story")
        items = [NewsItem(title=r.title, outlet=r.outlet, link=r.url, published_at=r.published_at)
                 for r in db.query(UFCNewsTopicItem).filter(UFCNewsTopicItem.topic_id == t.id)]
        fids = json.loads(t.fighter_ids or "[]")
        cluster = topics.Cluster(items=items, fighter_ids=fids, outlets=sorted({i.outlet for i in items if i.outlet}),
                                 newest=max((i.published_at for i in items if i.published_at), default=None),
                                 score=t.score)
        meter = llm.Meter(persist=True, month_spent=llm.month_spent(db))
        lab = {"title": t.title, "kind": t.kind, "follow_up": None}
        story = write_one(db, meter, entities.build(db), cluster, lab, True, t)
        from app.services import response_cache
        response_cache.invalidate()
        return {k: v for k, v in story.items() if k != "draft"}
    finally:
        db.close()
