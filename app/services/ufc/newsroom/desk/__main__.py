import argparse
import json
import logging
from pathlib import Path

from app.services.ufc.newsroom.desk import desk

p = argparse.ArgumentParser(prog="python -m app.services.ufc.newsroom.desk", description=desk.__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dry-run", action="store_true", help="write nothing to the database")
p.add_argument("--top", type=int, default=3)
p.add_argument("--out", type=Path, default=None)
p.add_argument("--discover-only", action="store_true", help="list the stories, write none")
a = p.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
for noisy in ("httpx", "httpx2", "trafilatura"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
s = desk.run(persist=not a.dry_run, top=a.top, out_dir=a.out, discover_only=a.discover_only)
print(json.dumps({"usd": s["usd"], "updated": s["updated"],
                  "stories": [{k: v for k, v in x.items() if k != "draft"} for x in s["stories"]]},
                 indent=2, default=str))
