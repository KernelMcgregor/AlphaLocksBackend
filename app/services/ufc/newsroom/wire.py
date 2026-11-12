"""The wire desk: free news about the two fighters in a bout.

Searches are built from the database, never typed by hand. For each fight:

    matchup     "Red Name" "Blue Name"            since the bout could have been booked
    recent      "Name"                            since that fighter's last fight
    background  "Name" UFC                        only while their storyline ledger is thin

Google News RSS covers most outlets (MMA Fighting, Sherdog, ESPN, Yahoo, MMA Junkie...).
Headlines are scored before anything is fetched, so only the few best stories per search
cost a page fetch and a DeepSeek extraction.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import logging
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import httpx

from app.services.ufc.newsroom.schemas import Article, NewsItem

log = logging.getLogger(__name__)

GOOGLE_NEWS = "https://news.google.com/rss/search"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
#: Pages built by script (nothing to read without a browser) or that refuse fetches.
SKIP_DOMAINS = ("espn.com", "youtube.com", "x.com", "twitter.com", "instagram.com",
                "tiktok.com", "facebook.com", "reddit.com", "tapology.com")
#: Video pages and listings, not stories.
SKIP_TITLE = re.compile(r"^(full fight|free fight|countdown|embedded|watch|live|ufc \d+ (live|results))\b", re.I)
MIN_TEXT = 500

LIMITS = {"matchup": 8, "recent": 4, "background": 3}


def _date(s: str | None) -> dt.date | None:
    if not s:
        return None
    try:
        return email.utils.parsedate_to_datetime(s).date()
    except Exception:
        return None


def google_news(query: str, timeout: float = 20.0) -> list[NewsItem]:
    try:
        r = httpx.get(GOOGLE_NEWS, params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                      timeout=timeout, headers={"User-Agent": UA})
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        log.warning(f"Google News failed for {query!r}: {e}")
        return []
    out = []
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        outlet = (it.findtext("source") or "").strip() or None
        # Google appends " - Outlet" to every title.
        if outlet and title.endswith(f" - {outlet}"):
            title = title[: -len(outlet) - 3]
        out.append(NewsItem(title=title, outlet=outlet, link=it.findtext("link") or "",
                            published_at=_date(it.findtext("pubDate")), query=query))
    return out


# ---------------------------------------------------------------------------
# Searches and headline scoring
# ---------------------------------------------------------------------------

def queries_for(red: dict, blue: dict, today: dt.date | None = None) -> list[tuple[str, str]]:
    """(query, tier) pairs for a bout. `red`/`blue`: {name, last_fight, ledger_size}."""
    today = today or dt.date.today()
    booked_since = today - dt.timedelta(days=120)
    qs = [(f'"{red["name"]}" "{blue["name"]}" after:{booked_since.isoformat()}', "matchup")]
    for f in (red, blue):
        since = f.get("last_fight") or (today - dt.timedelta(days=180))
        qs.append((f'"{f["name"]}" after:{(since + dt.timedelta(days=1)).isoformat()}', "recent"))
        if (f.get("ledger_size") or 0) < 3:
            qs.append((f'"{f["name"]}" UFC', "background"))
    return qs


def _mentions(text: str, name: str) -> float:
    """1.5 for the full name, 1 for the surname alone (case-sensitive, so "Van" the
    fighter is not "van" the vehicle), else 0."""
    if name.lower() in text.lower():
        return 1.5
    last = name.split()[-1]
    return 1.0 if re.search(rf"\b{re.escape(last)}\b", text) else 0.0


def score(item: NewsItem, names: list[str], tier: str, today: dt.date) -> float:
    if SKIP_TITLE.search(item.title):
        return -1
    host = urlparse(item.link).hostname or ""
    if any(d in host for d in SKIP_DOMAINS) or (item.outlet or "").lower() == "espn":
        return -1
    hits = [_mentions(item.title, n) for n in names]
    s = sum(hits) + (2.0 if all(hits) else 0.0)
    if tier != "background" and item.published_at:
        age = (today - item.published_at).days
        s += max(0.0, 1.5 - age / 30)
    return s


def pick_headlines(results: dict[str, list[NewsItem]], tiers: dict[str, str],
                   names: list[str], today: dt.date | None = None) -> list[tuple[NewsItem, str]]:
    """Best headlines per search, deduplicated across searches by title."""
    today = today or dt.date.today()
    seen, chosen = set(), []
    for q, items in results.items():
        tier = tiers[q]
        ranked = sorted(items, key=lambda i: score(i, names, tier, today), reverse=True)
        n = 0
        for it in ranked:
            key = re.sub(r"\W+", " ", it.title.lower()).strip()[:80]
            if key in seen or score(it, names, tier, today) <= 0:
                continue
            seen.add(key)
            chosen.append((it, tier))
            n += 1
            if n >= LIMITS[tier]:
                break
    return chosen


# ---------------------------------------------------------------------------
# Fetching a story
# ---------------------------------------------------------------------------

TWEET = re.compile(r"https?://(?:www\.)?(?:twitter|x)\.com/([A-Za-z0-9_]{1,15})/status/(\d+)")
YT_EMBED = re.compile(r"(?:youtube(?:-nocookie)?\.com/embed/|youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})")


def resolve(link: str) -> str | None:
    """A Google News link to the publisher's own URL."""
    if "news.google.com" not in link:
        return link
    try:
        from googlenewsdecoder import gnewsdecoder
        d = gnewsdecoder(link, interval=None)
        return d.get("decoded_url") if d.get("success") else None
    except Exception as e:
        log.warning(f"Could not decode {link[:80]}: {e}")
        return None


def embeds(html: str) -> tuple[list[str], list[str]]:
    tweets = list(dict.fromkeys(f"https://x.com/{u}/status/{i}" for u, i in TWEET.findall(html)))
    videos = list(dict.fromkeys(f"https://www.youtube.com/watch?v={v}" for v in YT_EMBED.findall(html)))
    return tweets[:6], videos[:4]


def fetch(url: str, item: NewsItem | None = None, timeout: float = 20.0) -> Article | None:
    import trafilatura

    host = urlparse(url).hostname or ""
    if any(d in host for d in SKIP_DOMAINS):
        return None
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": UA})
        if r.status_code >= 400:
            log.info(f"{r.status_code} for {url}")
            return None
        html = r.text
    except Exception as e:
        log.info(f"Fetch failed for {url}: {e}")
        return None
    text = trafilatura.extract(html, include_comments=False, include_tables=False, favor_precision=True) or ""
    if len(text) < MIN_TEXT:
        return None
    meta = trafilatura.extract_metadata(html)
    published = item.published_at if item else None
    if meta and meta.date and not published:
        try:
            published = dt.date.fromisoformat(meta.date[:10])
        except ValueError:
            pass
    tweets, videos = embeds(html)
    return Article(
        url=str(r.url), outlet=(item.outlet if item else None) or (meta.sitename if meta else None) or host,
        title=(item.title if item else None) or (meta.title if meta else None),
        author=meta.author if meta else None, published_at=published,
        text=text[:20000], tweets=tweets, videos=videos,
    )


def gather(red: dict, blue: dict, today: dt.date | None = None) -> list[tuple[NewsItem, str, str]]:
    """Search, score and resolve: (headline, tier, publisher url) for the stories worth
    reading. Fetching is left to the caller so stories already stored are not refetched."""
    today = today or dt.date.today()
    qs = queries_for(red, blue, today)
    results, tiers = {}, {}
    for q, tier in qs:
        results[q] = google_news(q)
        tiers[q] = tier
        time.sleep(0.5)
    picked = pick_headlines(results, tiers, [red["name"], blue["name"]], today)
    out = []
    for item, tier in picked:
        url = resolve(item.link)
        if not url:
            continue
        out.append((item, tier, url))
    log.info(f"Wire: {red['name']} vs {blue['name']}: {len(out)} stories from {sum(map(len, results.values()))} headlines")
    return out
