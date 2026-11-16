"""Right-facing fighter portraits, from ufc.com event pages.

A fighter's profile portrait (ufc_fighters.image_url / image_data) is the `_L_` pose,
facing left. The fight hero puts the blue corner on the right, where a left-facing
portrait looks away from the opponent. UFC.com shoots a mirrored `_R_` pose for blue
corners, but it is published only on event pages, never on athlete profiles:

    <div class="c-listing-fight__corner--blue"> ... <img src="https://ufc.com/images/styles/
      event_fight_card_upper_body_of_standing_athlete/s3/2026-07/STIRLING_NAVAJO_R_08-01.png?itok=..">

That image style is 185px wide; the original sits at the same path without the style
prefix (https://www.ufc.com/images/2026-07/STIRLING_NAVAJO_R_08-01.png, ~1000x3300 PNG).
This downloads it, shrinks it to MAX_HEIGHT keeping the alpha channel, and stores it in
ufc_fighters.image_right_*; GET /ufc/fighters/{id}/image?pose=right serves it.

Any `_R_` file on a card counts, red corner or blue: the pose belongs to the fighter.
An image is stored only when its filename names the fighter it is matched to.

    python -m app.services.ufc.fighter_poses --dry-run              # fetch + report, no writes
    python -m app.services.ufc.fighter_poses                        # upcoming events
    python -m app.services.ufc.fighter_poses --include-past-days 14 # plus recent cards
    python -m app.services.ufc.fighter_poses --refresh              # re-download stored ones
"""

from __future__ import annotations

import argparse
import datetime as dt
import html as htmllib
import io
import logging
import re
import time
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

import httpx
from bs4 import BeautifulSoup

from app.services.ufc.event_times import fetch_listing, match_card
from app.services.ufc.ufc_profile_scraper import _fighter_slug, _normalize, image_names_fighter

log = logging.getLogger("fighter_poses")

EVENT_URL = "https://www.ufc.com/event/{slug}"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
#: Stored portrait height. The hero renders well below this even on retina screens.
MAX_HEIGHT = 900
#: Above this, a full-colour PNG is re-encoded as a 256-colour palette PNG (alpha kept).
TARGET_BYTES = 600 * 1024
#: Originals run ~4-5MB; anything far above that is not a portrait.
MAX_SOURCE_BYTES = 25 * 1024 * 1024
REQUEST_GAP_S = 1.0

_RIGHT_RE = re.compile(r"_R(?:_[^/]*)?\.(?:png|jpe?g|webp)$", re.I)


# ---------------------------------------------------------------------------
# Parsing (pure)
# ---------------------------------------------------------------------------

@dataclass
class Corner:
    corner: str                     # "red" | "blue"
    name: str                       # display name, e.g. "Jiří Procházka"
    athlete_slug: str | None        # "jiri-prochazka"
    image_src: str | None


@dataclass
class Bout:
    red: Corner
    blue: Corner

    @property
    def corners(self) -> tuple[Corner, Corner]:
        return self.red, self.blue


def _corner(fight, color: str) -> Corner:
    img = fight.select_one(f".c-listing-fight__corner-image--{color} img")
    name_el = fight.select_one(f".c-listing-fight__corner-name--{color}")
    link = (name_el.select_one("a[href]") if name_el else None) or fight.select_one(
        f".c-listing-fight__corner-image--{color} a[href]")
    slug = None
    if link is not None:
        m = re.search(r"/athlete/([^/?#]+)", link.get("href", ""))
        slug = m.group(1) if m else None
    name = " ".join(name_el.get_text(" ").split()) if name_el else ""
    src = img.get("src") if img is not None else None
    return Corner(color, htmllib.unescape(name), slug, htmllib.unescape(src) if src else None)


def parse_event_page(page: str) -> list[Bout]:
    """Every bout on a ufc.com event page, with both corners' name, slug and image."""
    soup = BeautifulSoup(page, "html.parser")
    return [Bout(_corner(f, "red"), _corner(f, "blue")) for f in soup.select(".c-listing-fight")]


def _basename(src: str) -> str:
    return unquote(urlsplit(src).path).rsplit("/", 1)[-1]


def is_right_pose(src: str | None) -> bool:
    """Whether an image is a `_R_` (right-facing) file, e.g. STIRLING_NAVAJO_R_08-01.png."""
    return bool(src) and bool(_RIGHT_RE.search(_basename(src)))


def full_size_url(src: str) -> str:
    """The original behind a Drupal image-style URL.

    https://ufc.com/images/styles/<style>/s3/2026-07/X_R_08-01.png?itok=..
      -> https://www.ufc.com/images/2026-07/X_R_08-01.png
    """
    path = urlsplit(src).path
    m = re.search(r"/images/styles/[^/]+/s3/(.+)$", path) or re.search(r"/images/(.+)$", path)
    rest = m.group(1) if m else path.lstrip("/")
    return f"https://www.ufc.com/images/{rest}"


def _fold(s: str | None) -> str:
    return re.sub(r"[^a-z]", "", _normalize(s or ""))


def match_fighter(corner: Corner, fighters: list) -> object | None:
    """Which of an event's fighters a ufc.com corner is, or None.

    Full name first, then the athlete-page slug, then the image filename (unique among
    the card's fighters, with the surname also in the display name or slug).
    """
    target = _fold(corner.name)
    for f in fighters:
        if target and _fold(f"{f.first_name} {f.last_name}") == target:
            return f
    if corner.athlete_slug:
        for f in fighters:
            if _fighter_slug(f.first_name or "", f.last_name or "") == corner.athlete_slug:
                return f
    if corner.image_src:
        base = _basename(corner.image_src)
        hay = target + _fold(corner.athlete_slug)
        hits = [f for f in fighters
                if image_names_fighter(base, f.first_name, f.last_name)
                and _fold(f.last_name) and _fold(f.last_name) in hay]
        if len(hits) == 1:
            return hits[0]
    return None


@dataclass
class Plan:
    """What one pass would store: fighter id -> (name, full-size URL, event slug)."""
    store: dict = field(default_factory=dict)
    no_right_bouts: list = field(default_factory=list)   # bouts with no `_R_` file at all
    unmatched: list = field(default_factory=list)        # `_R_` files not matched to a fighter
    already: list = field(default_factory=list)          # skipped: already stored
    events: list = field(default_factory=list)


def plan_event(bouts: list[Bout], fighters: list, plan: Plan, slug: str,
               refresh: bool = False, roster_lookup=None) -> None:
    """Fold one event page's bouts into `plan`. Pure apart from mutating `plan`.

    `roster_lookup(corner)` is the fallback for a bout ufc.com lists that our card lacks:
    it returns the roster's one fighter with that exact name, or None.
    """
    for b in bouts:
        rights = [c for c in b.corners if is_right_pose(c.image_src)]
        if not rights:
            plan.no_right_bouts.append(f"{slug}: {b.red.name} vs {b.blue.name}")
            continue
        for c in rights:
            f = match_fighter(c, fighters)
            if f is None and roster_lookup is not None:
                f = roster_lookup(c)
            if f is None or not image_names_fighter(_basename(c.image_src), f.first_name, f.last_name):
                plan.unmatched.append(f"{slug}: {c.name} ({_basename(c.image_src)})")
                continue
            if f.id in plan.store:
                continue
            if getattr(f, "image_right_url", None) and not refresh:
                plan.already.append(f"{f.first_name} {f.last_name}")
                continue
            plan.store[f.id] = (f"{f.first_name} {f.last_name}", full_size_url(c.image_src), slug)


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

def shrink(data: bytes, max_height: int = MAX_HEIGHT) -> tuple[bytes, str]:
    """Height-capped PNG with alpha kept; a palette PNG when the full-colour one is big."""
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P", "PA") else "RGB")
    if img.height > max_height:
        img = img.resize((max(1, round(img.width * max_height / img.height)), max_height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    out = buf.getvalue()
    if len(out) > TARGET_BYTES:
        q = img.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
        buf = io.BytesIO()
        q.save(buf, format="PNG", optimize=True)
        if len(buf.getvalue()) < len(out):
            out = buf.getvalue()
    return out, "image/png"


def download(client: httpx.Client, url: str) -> tuple[bytes, str]:
    resp = client.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    mime = (resp.headers.get("content-type") or "").split(";")[0].strip()
    if not mime.startswith("image/"):
        raise RuntimeError(f"not an image ({mime or 'no content-type'})")
    if not resp.content or len(resp.content) > MAX_SOURCE_BYTES:
        raise RuntimeError(f"bad size ({len(resp.content)} bytes)")
    return shrink(resp.content)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def pose_columns_exist(db) -> bool:
    from sqlalchemy import inspect

    from app.config import settings

    schema = None if settings.DATABASE_URL.startswith("sqlite") else "ufc"
    have = {c["name"] for c in inspect(db.get_bind()).get_columns("ufc_fighters", schema=schema)}
    return "image_right_url" in have


def _event_fighters(db, event_id: int, with_right: bool) -> list:
    """The event's fighters as light rows (no image bytes): id, names, image_right_url."""
    from app.models.ufc import UFCFight, UFCFighter

    fights = db.query(UFCFight.red_fighter_id, UFCFight.blue_fighter_id).filter(UFCFight.event_id == event_id).all()
    ids = {i for pair in fights for i in pair if i}
    if not ids:
        return []
    cols = [UFCFighter.id, UFCFighter.first_name, UFCFighter.last_name]
    if with_right:
        cols.append(UFCFighter.image_right_url)
    return db.query(*cols).filter(UFCFighter.id.in_(ids)).all()


def _roster_lookup(db, with_right: bool):
    """Fallback matcher: the roster's unique fighter whose full name equals the corner's."""
    from sqlalchemy import func

    from app.models.ufc import UFCFighter

    def lookup(corner: Corner):
        words = corner.name.split()
        if len(words) < 2:
            return None
        cols = [UFCFighter.id, UFCFighter.first_name, UFCFighter.last_name]
        if with_right:
            cols.append(UFCFighter.image_right_url)
        # ilike on the folded last word keeps the scan small; the exact check is below.
        rows = db.query(*cols).filter(func.lower(UFCFighter.last_name).like(
            f"%{_normalize(words[-1])[-4:]}%")).all()
        target = _fold(corner.name)
        hits = [r for r in rows if _fold(f"{r.first_name} {r.last_name}") == target]
        return hits[0] if len(hits) == 1 else None
    return lookup


def run(dry_run: bool = False, refresh: bool = False, include_past_days: int = 0) -> dict:
    from app.database import SessionLocal
    from app.models.ufc import UFCEvent, UFCFighter

    client = httpx.Client(headers={"User-Agent": USER_AGENT, "Referer": "https://www.ufc.com/"},
                          follow_redirects=True, timeout=30.0)
    db = SessionLocal()
    plan = Plan()
    failed: list[str] = []
    saved = 0
    try:
        with_right = pose_columns_exist(db)
        if not with_right and not dry_run:
            raise SystemExit("ufc_fighters has no image_right_* columns: "
                             "run `python -m scripts.add_fighter_pose_columns` first")
        cards = fetch_listing(client)
        since = dt.date.today() - dt.timedelta(days=max(include_past_days, 1))
        events = db.query(UFCEvent).filter(UFCEvent.date >= since).order_by(UFCEvent.date).all()
        log.info(f"{len(events)} events since {since}; {len(cards)} cards on the ufc.com listing")

        for ev in events:
            card = match_card(ev.name, ev.date, cards)
            if card is None:
                log.warning(f"  no ufc.com card for {ev.name} ({ev.date})")
                failed.append(f"event {ev.name}: no ufc.com card")
                continue
            time.sleep(REQUEST_GAP_S)
            try:
                resp = client.get(EVENT_URL.format(slug=card.slug))
                resp.raise_for_status()
            except Exception as exc:
                log.warning(f"  {card.slug}: {exc}")
                failed.append(f"event {card.slug}: {exc}")
                continue
            bouts = parse_event_page(resp.text)
            before = len(plan.store)
            plan_event(bouts, _event_fighters(db, ev.id, with_right), plan, card.slug, refresh=refresh,
                       roster_lookup=_roster_lookup(db, with_right))
            plan.events.append(f"{ev.name} -> /event/{card.slug}: {len(bouts)} bouts, "
                               f"{len(plan.store) - before} to store")
            log.info("  " + plan.events[-1])

        for fid, (name, url, slug) in plan.store.items():
            if dry_run:
                log.info(f"  would store {name}: {url}")
                continue
            time.sleep(REQUEST_GAP_S)
            try:
                data, mime = download(client, url)
            except Exception as exc:
                log.warning(f"  {name}: {url}: {exc}")
                failed.append(f"{name}: {exc}")
                continue
            db.query(UFCFighter).filter(UFCFighter.id == fid).update({
                "image_right_url": url,
                "image_right_data": data,
                "image_right_mime": mime,
                "image_right_fetched_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
            }, synchronize_session=False)
            db.commit()
            saved += 1
            log.info(f"  stored {name}: {len(data) // 1024}KB")
    finally:
        db.close()
        client.close()

    for line in plan.no_right_bouts:
        log.info(f"  no _R_ file: {line}")
    for line in plan.unmatched:
        log.warning(f"  unmatched _R_ file: {line}")
    return {
        "dry_run": dry_run,
        "events": plan.events,
        "to_store": len(plan.store),
        "would_store": [f"{n} ({fid})" for fid, (n, _, _) in plan.store.items()],
        "saved": saved,
        "already_stored": len(plan.already),
        "bouts_without_right_file": len(plan.no_right_bouts),
        "unmatched": plan.unmatched,
        "failed": failed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="fetch pages and report; write nothing")
    ap.add_argument("--refresh", action="store_true", help="re-download fighters already stored")
    ap.add_argument("--include-past-days", type=int, default=0,
                    help="also cover events up to N days back (default: upcoming only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import json
    print(json.dumps(run(dry_run=args.dry_run, refresh=args.refresh,
                         include_past_days=args.include_past_days), indent=1, default=str))


if __name__ == "__main__":
    main()
