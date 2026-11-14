"""The day's headlines, from free sources only.

Three layers of searches, rebuilt every run; nobody types topics in:

  1. blind sweeps        `UFC when:1d`, `MMA when:1d`, plus nets by kind of story
                         (rumors, drama, injuries, contracts)
  2. who matters now     names from the DB: champions, our top five per division,
                         fighters booked in the next 30 days, fighters who fought in
                         the last 14 days, packed into `"A" OR "B" OR ...` queries
  3. follow-ups          names given by the caller: fighters whose mentions spiked and
                         fighters in stories still developing

Plus the MMA Fighting and Sherdog feeds. Only headlines are read here; pages are
fetched later, for the few stories that get written.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import time
import xml.etree.ElementTree as ET

import httpx

from app.services.ufc.newsroom.schemas import NewsItem
from app.services.ufc.newsroom.wire import UA, _date, google_news

log = logging.getLogger(__name__)

SWEEPS = [
    "UFC when:1d",
    "MMA when:1d",
    'UFC (rumor OR rumored OR "in talks" OR targeted OR "set to") when:1d',
    'UFC (callout OR "calls out" OR beef OR brawl OR attacked) when:1d',
    'UFC (injury OR injured OR "pulls out" OR withdraws OR surgery) when:1d',
    'UFC (signs OR "re-signs" OR released OR retires OR retirement) when:1d',
]
FEEDS = {
    "MMA Fighting": "https://www.mmafighting.com/rss/current.xml",
    "Sherdog": "https://www.sherdog.com/rss/news.xml",
}
NAMES_PER_QUERY = 10


def watch_names(db, today: dt.date | None = None) -> list[str]:
    """Layer 2: the fighters who matter right now, from our own tables."""
    from sqlalchemy import or_
    from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFighterRanking
    from app.services.ufc.champions import current_champions

    today = today or dt.date.today()
    ids = set(current_champions(db).values())
    ids |= {fid for (fid,) in db.query(UFCFighterRanking.fighter_id).filter(UFCFighterRanking.rank <= 5)}
    for red, blue in (db.query(UFCFight.red_fighter_id, UFCFight.blue_fighter_id)
                      .join(UFCEvent, UFCEvent.id == UFCFight.event_id)
                      .filter(UFCEvent.date >= today - dt.timedelta(days=14),
                              UFCEvent.date <= today + dt.timedelta(days=30))):
        ids.update((red, blue))
    rows = db.query(UFCFighter.first_name, UFCFighter.last_name).filter(UFCFighter.id.in_(ids)).all() if ids else []
    return sorted({f"{a} {b}".strip() for a, b in rows if b})


def name_queries(names: list[str], window: str = "1d") -> list[str]:
    out = []
    for i in range(0, len(names), NAMES_PER_QUERY):
        chunk = names[i:i + NAMES_PER_QUERY]
        out.append(" OR ".join(f'"{n}"' for n in chunk) + f" when:{window}")
    return out


def outlet_feed(outlet: str, url: str, timeout: float = 20.0) -> list[NewsItem]:
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": UA})
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        log.warning(f"{outlet} feed failed: {e}")
        return []
    out = []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for it in root.findall(".//item"):
        out.append(NewsItem(title=(it.findtext("title") or "").strip(), outlet=outlet,
                            link=(it.findtext("link") or "").strip(),
                            published_at=_date(it.findtext("pubDate")), query=f"rss:{outlet}"))
    for e in root.findall(".//a:entry", ns):
        link = e.find("a:link", ns)
        published = (e.findtext("a:published", namespaces=ns) or e.findtext("a:updated", namespaces=ns) or "")[:10]
        out.append(NewsItem(title=(e.findtext("a:title", namespaces=ns) or "").strip(), outlet=outlet,
                            link=link.get("href") if link is not None else "",
                            published_at=dt.date.fromisoformat(published) if published else None,
                            query=f"rss:{outlet}"))
    return out


def _key(title: str) -> str:
    return re.sub(r"\W+", " ", title.lower()).strip()[:90]


def harvest(db, follow_names: list[str] | None = None, today: dt.date | None = None,
            pause: float = 0.5) -> list[NewsItem]:
    """Every headline from the three layers, deduplicated by title."""
    today = today or dt.date.today()
    queries = list(SWEEPS) + name_queries(watch_names(db, today))
    if follow_names:
        queries += name_queries(sorted(set(follow_names)), window="2d")
    seen, items = set(), []

    def add(batch):
        for it in batch:
            if not it.title or not it.link:
                continue
            if it.published_at and (today - it.published_at).days > 3:
                continue
            k = _key(it.title)
            if k in seen:
                continue
            seen.add(k)
            items.append(it)

    for q in queries:
        batch = google_news(q)
        if not batch:  # one retry: Google News throttles bursts now and then
            time.sleep(2)
            batch = google_news(q)
        add(batch)
        time.sleep(pause)
    for outlet, url in FEEDS.items():
        add(outlet_feed(outlet, url))
    log.info(f"Harvest: {len(items)} headlines from {len(queries)} searches + {len(FEEDS)} feeds")
    return items
