"""Amateur records from the local Sherdog page cache. No requests, no DB writes.

The scraper parses amateur tables but drops them (parse_fighter_page(include_amateur=False)),
so the DB has pro bouts only. This re-reads data/sherdog/cache/fighter_<id>.html for every
Sherdog fighter linked to a UFC fighter and writes their amateur bouts to
data/feature_wf/sherdog_amateur.pkl (sherdog_id, date, result, method_class).

Caveat: pages were cached ~2026-09-30; amateur careers end before the pro debut, so that
staleness doesn't matter here.

    PYTHONPATH=. venv/bin/python -m scripts.sherdog_amateur_parse
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.services.ufc.sherdog_scraper import parse_fighter_page

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "sherdog" / "cache"
OUT = ROOT / "data" / "feature_wf" / "sherdog_amateur.pkl"


def main():
    from sqlalchemy import text
    from app.database import engine
    ids = [r[0] for r in engine.connect().execute(text(
        "SELECT sherdog_id FROM ufc.sherdog_fighters WHERE ufc_fighter_id IS NOT NULL")).fetchall()]
    rows, pages, missing = [], 0, 0
    for sid in ids:
        f = CACHE / f"fighter_{sid}.html"
        if not f.exists():
            missing += 1
            continue
        pages += 1
        _, bouts = parse_fighter_page(f.read_text(errors="ignore"), f"https://www.sherdog.com/fighter/x-{sid}",
                                      include_amateur=True)
        rows += [(sid, b.date, b.result, b.method_class) for b in bouts if b.section == "amateur"]
    df = pd.DataFrame(rows, columns=["sherdog_id", "date", "result", "method_class"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT)
    print(f"{len(ids)} linked fighters, {pages} cached pages ({missing} missing), "
          f"{df.sherdog_id.nunique()} with amateur bouts, {len(df)} amateur bouts -> {OUT}")


if __name__ == "__main__":
    main()
