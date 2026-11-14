"""What the news desk remembers between runs.

    mentions     a day's headline count per fighter (the spike signal)
    follow_names who to search for next run: spiking fighters and fighters in stories
                 still developing
    match_topic  is a new cluster a story we are already running? (fighters + wording)
    save_story   the topic, its headlines, the articles read and their claims
    cover        the best lead photo of the articles read, re-encoded and stored
    dossier      a short running summary per fighter, from their ledger
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re

import httpx
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.services.ufc.newsroom import llm
from app.services.ufc.newsroom.desk.topics import BIG, Cluster
from app.services.ufc.newsroom.schemas import Article, Claim

log = logging.getLogger(__name__)

DEVELOPING = {"incident", "rumored_fight", "injury", "controversy", "callout", "trash_talk"}
SAME_TOPIC_SIM = 0.2
REWRITE_NEW_OUTLETS = 3


def _now() -> dt.datetime:
    return dt.datetime.utcnow().replace(microsecond=0)


# ---------------------------------------------------------------------------
# Mentions and follow-ups
# ---------------------------------------------------------------------------

def record_mentions(db: Session, counts: dict[int, int], day: dt.date | None = None) -> None:
    from app.models.ufc import UFCNewsMention
    day = day or dt.date.today()
    for fid, n in counts.items():
        row = db.query(UFCNewsMention).filter(UFCNewsMention.fighter_id == fid, UFCNewsMention.day == day).first()
        if row is None:
            db.add(UFCNewsMention(fighter_id=fid, day=day, count=n))
        else:
            row.count = max(row.count, n)  # two runs a day: keep the fuller count
    db.commit()


def follow_names(db: Session, names: dict[int, str], today: dt.date | None = None) -> list[str]:
    """Fighters to search for by name this run: today's count well above their 14-day
    norm, and fighters in stories from the last week that are still developing."""
    from app.models.ufc import UFCNewsMention, UFCNewsTopic
    today = today or dt.date.today()
    out: set[int] = set()
    since = today - dt.timedelta(days=14)
    recent = dict(db.query(UFCNewsMention.fighter_id, UFCNewsMention.count)
                  .filter(UFCNewsMention.day >= today - dt.timedelta(days=1)).all())
    base = dict(db.query(UFCNewsMention.fighter_id, func.sum(UFCNewsMention.count))
                .filter(UFCNewsMention.day >= since, UFCNewsMention.day < today - dt.timedelta(days=1))
                .group_by(UFCNewsMention.fighter_id).all())
    for fid, n in recent.items():
        norm = (base.get(fid) or 0) / 12
        if n >= 3 and n >= 3 * max(norm, 1):
            out.add(fid)
    cutoff = _now() - dt.timedelta(days=3)
    for t in db.query(UFCNewsTopic).filter(UFCNewsTopic.last_seen >= cutoff,
                                           UFCNewsTopic.kind.in_(DEVELOPING),
                                           UFCNewsTopic.status.in_(("published", "needs_review"))):
        out.update(json.loads(t.fighter_ids or "[]")[:2])
    return sorted(names[f] for f in out if f in names)


# ---------------------------------------------------------------------------
# Topic memory
# ---------------------------------------------------------------------------

def match_topic(db: Session, cluster: Cluster, days: int = 7):
    """The running story this cluster belongs to, or None."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from app.models.ufc import UFCNewsTopic, UFCNewsTopicItem

    since = _now() - dt.timedelta(days=days)
    topics = db.query(UFCNewsTopic).filter(UFCNewsTopic.last_seen >= since).all()
    if not topics:
        return None
    mine = set(cluster.fighter_ids)
    text = " ".join(cluster.headlines[:20])
    best, best_s = None, 0.0
    for t in topics:
        theirs = set(json.loads(t.fighter_ids or "[]"))
        shared = len(mine & theirs)
        if not shared and (mine or theirs):
            continue
        heads = [r[0] for r in db.query(UFCNewsTopicItem.title).filter(UFCNewsTopicItem.topic_id == t.id).limit(30)]
        other = " ".join([t.title, *heads])
        sim = float(cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform([text, other]))[0, 1])
        if (shared >= 2 or (shared and sim >= SAME_TOPIC_SIM) or sim >= 0.45) and sim > best_s:
            best, best_s = t, sim
    return best


def new_outlets(db: Session, topic, cluster: Cluster) -> int:
    from app.models.ufc import UFCNewsTopicItem
    have = {o for (o,) in db.query(UFCNewsTopicItem.outlet).filter(UFCNewsTopicItem.topic_id == topic.id)}
    return len(set(cluster.outlets) - have)


# ---------------------------------------------------------------------------
# Saving a story
# ---------------------------------------------------------------------------

def slugify(title: str, day: dt.date) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:70].strip("-")
    return f"{s}-{day:%Y%m%d}"


def attach_items(db: Session, topic, cluster: Cluster) -> None:
    from app.models.ufc import UFCNewsTopicItem
    have = {u for (u,) in db.query(UFCNewsTopicItem.url).filter(UFCNewsTopicItem.topic_id == topic.id)}
    for it in cluster.items:
        if it.link in have:
            continue
        have.add(it.link)
        db.add(UFCNewsTopicItem(topic_id=topic.id, url=it.link[:1000], outlet=(it.outlet or "")[:200],
                                title=it.title[:500], published_at=it.published_at))
    db.flush()
    topic.last_seen = _now()
    topic.outlet_count = (db.query(func.count(func.distinct(UFCNewsTopicItem.outlet)))
                          .filter(UFCNewsTopicItem.topic_id == topic.id).scalar() or 0)
    db.flush()


def save_story(db: Session, topic, cluster: Cluster, label: dict, fighters: list[dict], arts: list[Article],
               cs: list[Claim], draft: str, report: dict, status: str, focus: list[dict] | None = None,
               layout: str | None = None):
    """Create or update the topic, its headlines, the articles read and their claims."""
    from app.models.ufc import UFCNewsArticle, UFCNewsTopic, UFCResearchClaim

    now = _now()
    m = re.search(r"^#\s+(.+)$", draft or "", re.M)
    title = (m.group(1).strip() if m else label["title"])[:300]
    if topic is None:
        slug = slugify(title, dt.date.today())
        n = db.query(func.count(UFCNewsTopic.id)).filter(UFCNewsTopic.slug.like(f"{slug}%")).scalar() or 0
        topic = UFCNewsTopic(slug=slug if not n else f"{slug}-{n + 1}", title=title, kind=label["kind"],
                             first_seen=now, last_seen=now, status="draft")
        db.add(topic)
        db.flush()
    if status != "published" and topic.status == "published" and topic.content:
        # A live story whose rewrite failed the checker stays up as it was; the held
        # draft waits in the admin queue's report.
        report = {**report, "held_draft": draft}
        topic.report = json.dumps(report, default=str)
        attach_items(db, topic, cluster)
        db.commit()
        return topic
    topic.title, topic.kind = title, label["kind"]
    topic.fighter_ids = json.dumps([f["id"] for f in fighters])
    topic.focus_ids = json.dumps([f["id"] for f in (focus or fighters)])
    topic.layout = layout
    topic.score = cluster.score
    topic.content, topic.report = draft, json.dumps(report, default=str)
    if topic.status != "rejected":  # a story the editor killed stays dead
        topic.status = status
        if status == "published" and topic.published_at is None:
            topic.published_at = now
    attach_items(db, topic, cluster)

    by_url = {}
    for a in arts:
        row = db.query(UFCNewsArticle).filter(UFCNewsArticle.url == a.url[:1000]).first()
        if row is None:
            row = UFCNewsArticle(url=a.url[:1000], outlet=(a.outlet or "")[:200], title=(a.title or "")[:500],
                                 published_at=a.published_at, fetch_ok=True)
            db.add(row)
        row.text, row.author = a.text, (a.author or None) and a.author[:300]
        row.embeds = json.dumps({"tweets": a.tweets, "videos": a.videos})
        db.flush()
        by_url[a.url] = row.id
    for c in cs:
        dup = db.query(UFCResearchClaim.id).filter(UFCResearchClaim.fighter_id == c.fighter_id,
                                                   UFCResearchClaim.source_url == c.source_url,
                                                   UFCResearchClaim.text == c.text).first()
        if dup:
            continue
        db.add(UFCResearchClaim(
            topic_id=topic.id, article_id=by_url.get(c.source_url), mentions=json.dumps(c.mentions or [c.fighter_id]),
            **c.model_dump(exclude={"id", "source_text", "mentions", "topic_id"})))
    db.commit()
    return topic


# ---------------------------------------------------------------------------
# Cover photo
# ---------------------------------------------------------------------------

def cover(db: Session, topic, arts: list[Article]) -> bool:
    """The best lead photo of the articles read: big outlets first, screened by the
    preview covers' checks (size, shape, burned-in text) and framed on the faces."""
    from app.services.ufc.article_images import Candidate, analyse, _domain
    from app.services.ufc.newsroom.wire import UA

    if topic.image is not None:
        return False
    order = sorted([a for a in arts if a.image], key=lambda a: (a.outlet or "").lower() not in BIG)
    best = None
    with httpx.Client(timeout=20, follow_redirects=True, headers={"User-Agent": UA}) as client:
        for rank, a in enumerate(order[:6]):
            try:
                r = client.get(a.image)
                if r.status_code >= 400:
                    continue
            except Exception:
                continue
            c = analyse(Candidate(url=a.image, page=a.url, domain=_domain(a.url), title=a.title or "",
                                  query="og:image", rank=rank), r.content)
            if c.reject is None and (best is None or c.score > best[0].score):
                best = (c, a)
    if best is None:
        return False
    c, a = best
    topic.image, topic.image_focal_x, topic.image_focal_y = c.jpeg, c.focal_x, c.focal_y
    topic.image_source, topic.image_page = (a.outlet or c.domain)[:200], a.url[:1000]
    db.commit()
    return True


# ---------------------------------------------------------------------------
# Dossiers
# ---------------------------------------------------------------------------

DOSSIER = """You keep a running file on a UFC fighter for a news desk. From the sourced notes
below, write their current storylines in 3 to 5 short plain sentences, newest and most
important first: titles won or lost and how, rivalries and beef, injuries, legal or
contract news, what they are chasing next. Facts from the notes only, no stats, no
opinions. Return only the sentences."""


def refresh_dossier(db: Session, meter: llm.Meter, fighter_id: int, name: str) -> str | None:
    from app.models.ufc import UFCFighterDossier, UFCResearchClaim
    rows = (db.query(UFCResearchClaim)
            .filter(UFCResearchClaim.struck.is_(False),
                    (UFCResearchClaim.fighter_id == fighter_id) | UFCResearchClaim.mentions.like(f"%{fighter_id}%"))
            .order_by(UFCResearchClaim.newsworthiness.desc(), UFCResearchClaim.published_at.desc().nullslast())
            .limit(24).all())
    if len(rows) < 3:
        return None
    notes = "\n".join(f"- ({r.published_at or 'undated'}) {r.text}" for r in rows)
    summary = llm.chat(meter, "dossier", None, DOSSIER, f"Fighter: {name}\n\nNotes:\n{notes}", max_tokens=400).strip()
    if not summary:
        return None
    row = db.query(UFCFighterDossier).filter(UFCFighterDossier.fighter_id == fighter_id).first()
    if row is None:
        db.add(UFCFighterDossier(fighter_id=fighter_id, summary=summary))
    else:
        row.summary = summary
    db.commit()
    return summary
