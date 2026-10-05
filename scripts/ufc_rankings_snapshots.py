"""Archived UFC.com rankings, one snapshot per month, for fitting the P4P list.

UFC.com only shows the current rankings, so history comes from the Wayback Machine's
monthly captures of ufc.com/rankings. Each capture is parsed into every division's
champion and ranked list, including both pound-for-pound top 15s.

Output: data/ufc_rankings_history.json  {"YYYYMMDD": {"Men's Pound-for-Pound":
        {"champion": None, "ranked": [...]}, "Lightweight": {...}, ...}, ...}

Run: python -m scripts.ufc_rankings_snapshots [--from 2024] [--to 2026]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "ufc_rankings_history.json"
CDX = "https://web.archive.org/cdx/search/cdx"


def parse(page: str) -> dict:
    snap = {}
    for g in re.split(r'<div class="view-grouping-header">', page)[1:]:
        title = html.unescape(re.sub(r"<[^>]+>", " ", g[:g.find("</div>")]))
        title = re.sub(r"\s+", " ", title.split("Top Rank")[0]).strip()
        head = g[:g.find("<tbody>")] if "<tbody>" in g else ""
        champ = re.search(r"<h5>\s*<a[^>]*>([^<]+)</a>", head)
        ranked = re.findall(
            r'weight-class-rank">\s*\d+\s*</td>\s*<td class="views-field views-field-title">'
            r"\s*<a[^>]*>([^<]+)</a>", g)
        snap[title] = {
            "champion": html.unescape(champ.group(1)).strip() if champ else None,
            "ranked": [html.unescape(n).strip() for n in ranked],
        }
    return snap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2024")
    ap.add_argument("--to", dest="end", default="2026")
    a = ap.parse_args()

    rows = requests.get(CDX, params={
        "url": "ufc.com/rankings", "from": a.start, "to": a.end, "output": "json",
        "fl": "timestamp", "filter": "statuscode:200", "collapse": "timestamp:6",
    }, timeout=60).json()[1:]
    history = json.loads(OUT.read_text()) if OUT.exists() else {}
    for (ts,) in rows:
        key = ts[:8]
        if key in history:
            continue
        for attempt in range(3):
            try:
                r = requests.get(f"https://web.archive.org/web/{ts}id_/https://www.ufc.com/rankings",
                                 timeout=60)
                snap = parse(r.text)
                if any("Pound" in k for k in snap):
                    history[key] = snap
                    print(key, "ok")
                    break
            except requests.RequestException:
                pass
            time.sleep(5 * (attempt + 1))
        else:
            print(key, "no P4P table in capture; skipped")
        time.sleep(2)
    OUT.write_text(json.dumps(dict(sorted(history.items())), indent=1, ensure_ascii=False))
    print(f"{len(history)} snapshots -> {OUT}")


if __name__ == "__main__":
    main()
