import argparse
import json
import logging
from pathlib import Path

from app.services.ufc.newsroom import pipeline

p = argparse.ArgumentParser(prog="python -m app.services.ufc.newsroom", description=pipeline.__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--upcoming", action="store_true", help="top of every card within --days")
p.add_argument("--fight", type=int, action="append", help="one fight id (repeatable)")
p.add_argument("--days", type=int, default=None)
p.add_argument("--top-n", type=int, default=None)
p.add_argument("--force", action="store_true", help="rewrite even when nothing changed")
p.add_argument("--dry-run", action="store_true", help="write nothing to the database")
p.add_argument("--out", type=Path, default=None, help="write claims, drafts and reports here")
p.add_argument("--ensure-schema", action="store_true", help="create the newsroom tables")
a = p.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

if a.ensure_schema:
    print("Tables:", pipeline.ensure_schema())
if a.fight:
    print(json.dumps(pipeline.run(a.fight, persist=not a.dry_run, force=a.force, out_dir=a.out), indent=2, default=str))
elif a.upcoming:
    print(json.dumps(pipeline.run_upcoming(a.days, a.top_n, persist=not a.dry_run, force=a.force, out_dir=a.out),
                     indent=2, default=str))
