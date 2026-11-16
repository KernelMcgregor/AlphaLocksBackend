"""Clear portraits that belong to someone else, or are placeholders.

Before the scraper checked it (ufc_profile_scraper.image_names_fighter), a fighter with no
UFC.com hero portrait got the first image in the page body, usually an opponent's headshot
from the fight-results list (Heath Herring showed Brock Lesnar). This clears image_url and
the cached bytes on every fighter whose portrait URL names nobody in their own name, or is
a known placeholder, so the site shows the blank silhouette instead.

    PYTHONPATH=. venv/bin/python scripts/clear_wrong_portraits.py           # dry run
    PYTHONPATH=. venv/bin/python scripts/clear_wrong_portraits.py --apply   # writes
"""
from __future__ import annotations

import argparse
import re
from urllib.parse import unquote

from app.database import SessionLocal
from app.models.ufc import UFCFighter
from app.services.ufc.ufc_profile_scraper import PLACEHOLDER_IMAGE_MARKERS, image_names_fighter


# A filename that is only an id ('migrated_images/80BBEE01-1422-...jpg') names nobody, so it
# can't be judged; those are kept.
ANONYMOUS = re.compile(r"/[0-9A-Fa-f-]{16,}\.(png|jpe?g)$")


def wrong(f: UFCFighter) -> bool:
    url = f.image_url or ""
    if any(m in url for m in PLACEHOLDER_IMAGE_MARKERS):
        return True
    if ANONYMOUS.search(unquote(url).split("?")[0]):
        return False
    return not image_names_fighter(url, f.first_name, f.last_name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    db = SessionLocal()
    hits = [f for f in db.query(UFCFighter).filter(UFCFighter.image_url.isnot(None)) if wrong(f)]
    for f in hits[:40]:
        print(f"{f.first_name} {f.last_name}: {f.image_url.split('?')[0][-70:]}")
    print(f"{len(hits)} portraits to clear{'' if args.apply else ' (dry run)'}")
    if args.apply:
        for f in hits:
            f.image_url = None
            f.image_data = None
            f.image_mime = None
        db.commit()
        print("cleared")


if __name__ == "__main__":
    main()
