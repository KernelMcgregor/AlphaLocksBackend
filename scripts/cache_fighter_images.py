"""Download fighter headshots from `image_url` and store the bytes in the database.

Why store them at all: UFC.com serves portraits through Drupal image styles whose URLs
carry an `?itok=` signature. Those signatures rotate, so every URL in the database is on
a clock — the day one expires the portrait 404s and there is nothing to fall back on.
Caching the bytes decouples the site from that.

The companion endpoint is `GET /ufc/fighters/{id}/image`, which serves the cached bytes
when they exist and redirects to `image_url` when they do not, so pages can point at one
URL whether or not this script has run yet.

Safe to run in the background and safe to interrupt: work is selected by "has a URL, has
no bytes", committed in batches, and re-running picks up exactly where it stopped.

    # one-off, foreground
    ./venv/bin/python -m scripts.cache_fighter_images

    # the whole backlog, in the background, with a log to tail
    nohup ./venv/bin/python -m scripts.cache_fighter_images > /tmp/images.log 2>&1 &
    tail -f /tmp/images.log

Flags:
    --limit N         stop after N fighters (0 = all)
    --workers N       concurrent downloads (default 8)
    --refresh         re-download fighters that already have bytes
    --stale-days N    with --refresh, only re-download copies older than N days
    --clean-placeholders  null out `no-profile-image` URLs, then exit
    --dry-run         report what would be fetched, write nothing
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import text

try:
    from PIL import Image
except ImportError:  # optional — see _shrink
    Image = None

from app.config import settings
from app.database import SessionLocal
from app.models.ufc import UFCFighter

log = logging.getLogger("cache_images")

SCHEMA = "" if settings.DATABASE_URL.startswith("sqlite") else "ufc."
TABLE = f"{SCHEMA}ufc_fighters"

#: A headshot that does not fit here is not a headshot. UFC's full-body PNGs run well
#: under this; anything larger is a redirect to something unexpected.
MAX_BYTES = 8 * 1024 * 1024

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def ensure_columns() -> None:
    """Add the cache columns if they are not there yet.

    Idempotent DDL rather than a migration file: the three columns are additive and
    nullable, so there is nothing to roll back and nothing that can break a running app
    mid-deploy.
    """
    ddl = [
        f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS image_data BYTEA",
        f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS image_mime VARCHAR(40)",
        f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS image_fetched_at TIMESTAMP",
    ]
    if settings.DATABASE_URL.startswith("sqlite"):
        ddl = [s.replace("BYTEA", "BLOB") for s in ddl]

    db = SessionLocal()
    try:
        for stmt in ddl:
            try:
                db.execute(text(stmt))
                db.commit()
            except Exception as exc:  # column already exists on engines without IF NOT EXISTS
                db.rollback()
                log.debug(f"  ddl skipped: {exc}")
    finally:
        db.close()


def clean_placeholders() -> int:
    """Clear URLs that point at one of UFC.com's 'no portrait' stand-ins.

    Two kinds are in the database: the malformed 'https://www.ufc.com../themes/...
    no-profile-image.png' the profile scraper used to build, and 'SHADOW_Fighter_
    fullLength_RED.png', a perfectly valid URL for a black silhouette. Nulling both lets
    the UI fall back to initials instead of a broken image or a featureless body, and
    lets a later scrape fill in a real portrait.
    """
    db = SessionLocal()
    try:
        result = db.execute(
            text(
                f"UPDATE {TABLE} SET image_url = NULL "
                "WHERE image_url LIKE '%no-profile-image%' "
                "   OR image_url LIKE '%ufc.com..%' "
                "   OR image_url LIKE '%SHADOW_Fighter%'"
            )
        )
        db.commit()
        return result.rowcount or 0
    finally:
        db.close()


def _select(db, refresh: bool, stale_days: int, limit: int) -> list[UFCFighter]:
    q = db.query(UFCFighter).filter(
        UFCFighter.image_url.isnot(None),
        UFCFighter.image_url != "",
    )
    if refresh:
        if stale_days:
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=stale_days)
            q = q.filter(
                (UFCFighter.image_fetched_at.is_(None))
                | (UFCFighter.image_fetched_at < cutoff)
            )
    else:
        q = q.filter(UFCFighter.image_data.is_(None))

    # Ranked fighters first: they are the ones on the rankings rail and the fight cards,
    # so a partial run still fixes the portraits people actually see.
    q = q.order_by(UFCFighter.image_data.isnot(None), UFCFighter.last_name)
    if limit:
        q = q.limit(limit)
    return q.all()


def _shrink(data: bytes, max_width: int) -> tuple[bytes, str] | None:
    """Re-encode a portrait as a width-capped WebP.

    UFC's full-body PNGs average 286KB, so caching all 2787 of them verbatim would put
    ~0.76 GB in the database — far more than the site needs for images rendered at 90px.
    A 400px WebP holds up at every size the UI uses and costs ~2% of that. Returns None
    if Pillow is missing or the re-encode does not actually save anything, in which case
    the caller stores the original.
    """
    if Image is None or not max_width:
        return None
    try:
        from io import BytesIO

        img = Image.open(BytesIO(data))
        # Portraits are cut-outs — keep the alpha channel or they gain a black box.
        img = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")
        if img.width > max_width:
            height = round(img.height * max_width / img.width)
            img = img.resize((max_width, height), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="WEBP", quality=82, method=4)
        out = buf.getvalue()
        return (out, "image/webp") if out and len(out) < len(data) else None
    except Exception as exc:
        log.debug(f"  shrink failed, storing original: {exc}")
        return None


def _fetch(client: httpx.Client, url: str, max_width: int) -> tuple[bytes, str] | None:
    resp = client.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    mime = (resp.headers.get("content-type") or "").split(";")[0].strip()
    if not mime.startswith("image/"):
        raise RuntimeError(f"not an image ({mime or 'no content-type'})")
    data = resp.content
    if not data:
        raise RuntimeError("empty body")
    if len(data) > MAX_BYTES:
        raise RuntimeError(f"too large ({len(data) // 1024}KB)")
    return _shrink(data, max_width) or (data, mime)


def run(limit: int = 0, workers: int = 8, refresh: bool = False,
        stale_days: int = 0, dry_run: bool = False, max_width: int = 400) -> dict:
    ensure_columns()
    if max_width and Image is None:
        log.warning(
            "Pillow is not installed — storing source PNGs unresized. That is ~286KB per "
            "fighter (~0.76 GB for the full roster). `pip install Pillow` first, or pass "
            "--max-width 0 to accept it."
        )

    db = SessionLocal()
    try:
        targets = _select(db, refresh, stale_days, limit)
        log.info(f"{len(targets)} fighters to fetch ({'refresh' if refresh else 'missing only'})")
        if dry_run:
            for f in targets[:20]:
                log.info(f"  would fetch {f.first_name} {f.last_name}: {f.image_url[:80]}")
            return {"selected": len(targets), "saved": 0, "failed": 0}

        # (id, url) only — the ORM objects belong to this session and the download runs
        # on worker threads, which must not touch it.
        work = [(f.id, f.image_url) for f in targets]
    finally:
        db.close()

    client = httpx.Client(
        headers={"User-Agent": USER_AGENT, "Referer": "https://www.ufc.com/"},
        follow_redirects=True,
        timeout=30.0,
    )

    saved = failed = 0
    failures: list[str] = []
    db = SessionLocal()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            def job(item):
                fighter_id, url = item
                try:
                    return fighter_id, _fetch(client, url, max_width), None
                except Exception as exc:
                    return fighter_id, None, str(exc)

            for i, (fighter_id, result, err) in enumerate(pool.map(job, work), 1):
                if err or not result:
                    failed += 1
                    if len(failures) < 25:
                        failures.append(f"{fighter_id}: {err}")
                else:
                    data, mime = result
                    db.query(UFCFighter).filter(UFCFighter.id == fighter_id).update(
                        {
                            "image_data": data,
                            "image_mime": mime,
                            "image_fetched_at": datetime.now(timezone.utc).replace(tzinfo=None),
                        },
                        synchronize_session=False,
                    )
                    saved += 1

                # Batch commits so an interrupted run keeps everything before the last
                # boundary and the next run resumes from there.
                if i % 25 == 0:
                    db.commit()
                    log.info(f"  [{i}/{len(work)}] {saved} saved, {failed} failed")

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
        client.close()

    log.info(f"Done: {saved} saved, {failed} failed")
    for line in failures:
        log.info(f"  failed: {line}")
    return {"selected": len(work), "saved": saved, "failed": failed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--stale-days", type=int, default=0)
    ap.add_argument("--clean-placeholders", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-width", type=int, default=400,
                    help="downscale to this width as WebP before storing; 0 stores the original")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.clean_placeholders:
        ensure_columns()
        cleared = clean_placeholders()
        log.info(f"Cleared {cleared} placeholder image URLs")
        return

    run(
        limit=args.limit,
        workers=args.workers,
        refresh=args.refresh,
        stale_days=args.stale_days,
        dry_run=args.dry_run,
        max_width=args.max_width,
    )


if __name__ == "__main__":
    main()
