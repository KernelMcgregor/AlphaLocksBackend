"""BestFightOdds prop odds (method, winner x method, decision, round totals), parsed from
the event pages bfo_scraper already cached on disk (data/bfo/cache/pages).

Every event page carries prop rows (``<tr class="pr">``) in the same odds table as the
moneylines. Each priced cell has ``data-li="[book, side, matchup, prop_type, fighter]"``
and the book's LATEST price, which for a finished event is its closing line. Coverage
(checked on the cache): prop rows exist from 2012, but cells are only populated from
2021, when BFO's current books start; 2021+ covers ~500 fights a year.

Markets kept (keys use BFO's fighter order a/b; ``to_corners`` maps them to red/blue):
  wm_{a|b}_{ko|sub|dec}   "<fighter> wins by TKO/KO | submission | decision"  (6-way)
  dec_yes / dec_no        "Fight goes to decision" / "Fight doesn't go to decision"
  ou_{1.5|2.5|...}_{over|under}   total rounds

Consensus close = median implied probability across sportsbooks (exchange books excluded:
they trade in-play), then de-vigged within each group (the six winner x method cells; the
decision pair; each over/under pair) by simple normalisation.

Usage:
    python -m app.services.ufc.bfo_props --out data/bfo/props_close.csv [--load-db]
"""
from __future__ import annotations

import argparse
import glob
import html as _html
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.ufc.bfo_scraper import (
    DEFAULT_CACHE_DIR, _json_ld_date, american_to_prob, parse_american,
)

EXCHANGE_BOOKS = {28, 29}  # Polymarket, Kalshi
_ROW = re.compile(r'<tr class="pr[^"]*"[^>]*>(.*?)</tr>', re.S)
_CELL = re.compile(r'<td[^>]*data-li="(\[[^"]*\])"[^>]*>(.*?)</td>', re.S)
_TH = re.compile(r"<th[^>]*>(.*?)</th>", re.S)
_TAG = re.compile(r"<[^>]+>")
_WM = re.compile(r"wins by (TKO/KO|submission|decision)$")
_OU = re.compile(r"^(Over|Under) (\d+)(½)? rounds?$")
_METHOD = {"TKO/KO": "ko", "submission": "sub", "decision": "dec"}


def _label_key(label: str, fighter: int | None) -> str | None:
    m = _WM.search(label)
    if m and fighter in (1, 2):
        return f"wm_{'ab'[fighter - 1]}_{_METHOD[m.group(1)]}"
    if label == "Fight goes to decision":
        return "dec_yes"
    if label == "Fight doesn't go to decision":
        return "dec_no"
    m = _OU.match(label)
    if m:
        line = int(m.group(2)) + (0.5 if m.group(3) else 0.0)
        return f"ou_{line}_{m.group(1).lower()}"
    return None


def parse_props(html: str) -> list[dict]:
    """-> [{matchup, book, key, american}] for every priced prop cell we keep."""
    s = html.find('<table class="odds-table">')
    if s < 0:
        return []
    body = html[s:html.find("</table>", s)]
    out = []
    for row in _ROW.findall(body):
        th = _TH.search(row)
        if not th:
            continue
        label = " ".join(_html.unescape(_TAG.sub(" ", th.group(1))).split())
        for li, cell in _CELL.findall(row):
            li = json.loads(li)
            if len(li) != 5:  # "n/a" cells carry no book
                continue
            book, _side, matchup, _ptype, fighter = li
            key = _label_key(label, fighter)
            text = _html.unescape(_TAG.sub("", cell))
            price = parse_american(text.replace("▲", "").replace("▼", "").strip())
            if key and price is not None and book not in EXCHANGE_BOOKS:
                out.append({"matchup": matchup, "book": book, "key": key, "american": price})
    return out


def _devig_group(vals: dict, keys: list[str]) -> dict:
    present = [k for k in keys if k in vals]
    if len(present) != len(keys):
        return {}
    tot = sum(vals[k] for k in keys)
    return {k: vals[k] / tot for k in keys} if tot > 0 else {}


def consensus(rows: list[dict]) -> dict[int, dict]:
    """matchup -> {key: de-vigged consensus prob, 'n_books': int, 'overround_wm': float}."""
    by = defaultdict(lambda: defaultdict(list))
    books = defaultdict(set)
    for r in rows:
        p = american_to_prob(r["american"])
        if p is not None:
            by[r["matchup"]][r["key"]].append(p)
            books[r["matchup"]].add(r["book"])
    out = {}
    for mu, keys in by.items():
        med = {k: float(np.median(v)) for k, v in keys.items()}
        res = {"n_books": len(books[mu])}
        wm = [f"wm_{s}_{m}" for s in "ab" for m in ("ko", "sub", "dec")]
        if all(k in med for k in wm):
            res["overround_wm"] = sum(med[k] for k in wm)
        res.update(_devig_group(med, wm))
        res.update({f"{k}_nv": v for k, v in _devig_group(med, ["dec_yes", "dec_no"]).items()})
        if "dec_yes" in med and "dec_yes_nv" not in res:
            res["dec_yes_raw"] = med["dec_yes"]
        for k in med:
            if k.startswith("ou_") and k.endswith("_over"):
                line = k[3:-5]
                res.update({f"{kk}_nv": v for kk, v in
                            _devig_group(med, [f"ou_{line}_over", f"ou_{line}_under"]).items()})
        out[mu] = res
    return out


def corner_markets(c: dict, swapped: bool) -> dict[str, tuple[float, float | None]]:
    """One matchup's consensus -> {market: (prob, overround_of_group)} in red/blue terms.
    swapped: BFO fighter_a is the DB blue corner (bfo_scraper.match_matchups)."""
    a, b = ("blue", "red") if swapped else ("red", "blue")
    out = {}
    for m in ("ko", "sub", "dec"):
        for side, corner in (("a", a), ("b", b)):
            v = c.get(f"wm_{side}_{m}")
            if v is not None:
                out[f"{corner}_{m}"] = (v, c.get("overround_wm"))
    if c.get("dec_yes_nv") is not None:
        out["dec_yes"] = (c["dec_yes_nv"], None)
    for k, v in c.items():
        if k.startswith("ou_") and k.endswith("_over_nv"):
            out[k[:-3]] = (v, None)
    return out


def to_corners(cons: dict[int, dict], links: pd.DataFrame) -> pd.DataFrame:
    """links: bfo_matchup_id, db_fight_id, db_swapped (from data/bfo/odds.csv)."""
    link = links.dropna(subset=["db_fight_id"]).drop_duplicates("bfo_matchup_id")
    rows = []
    for mu, fid, swapped in zip(link["bfo_matchup_id"], link["db_fight_id"], link["db_swapped"]):
        c = cons.get(int(mu))
        if not c:
            continue
        r = {"fight_id": str(fid).split(".")[0], "bfo_matchup_id": int(mu), "n_books": c["n_books"],
             "overround_wm": c.get("overround_wm")}
        r.update({f"mkt_{k}": v for k, (v, _) in corner_markets(c, str(swapped) == "True").items()})
        rows.append(r)
    return pd.DataFrame(rows)


def backfill_closes(csv_path: str = "data/bfo/props_close.csv") -> int:
    """Load parsed closing props into ufc_prop_odds_history (source 'bfo_close',
    captured_at = event date). Idempotent: fights that already have bfo_close rows are skipped."""
    import datetime as _dt

    from app.database import SessionLocal
    from app.models.ufc import UFCEvent, UFCFight, UFCPropOddsHistory

    df = pd.read_csv(csv_path, dtype={"fight_id": str})
    db = SessionLocal()
    try:
        have = {f for (f,) in db.query(UFCPropOddsHistory.fight_id)
                .filter(UFCPropOddsHistory.source == "bfo_close").distinct()}
        dates = dict(db.query(UFCFight.id, UFCEvent.date).join(UFCEvent, UFCEvent.id == UFCFight.event_id))
        rows = []
        for r in df.to_dict("records"):
            fid = int(r["fight_id"])
            if fid in have or fid not in dates or dates[fid] is None:
                continue
            ts = _dt.datetime.combine(dates[fid], _dt.time())
            for k, v in r.items():
                if k.startswith("mkt_") and v == v and v is not None:
                    market = k[4:]
                    rows.append(UFCPropOddsHistory(
                        fight_id=fid, market=market, prob=round(float(v), 4),
                        n_books=int(r["n_books"]), source="bfo_close", captured_at=ts,
                        overround=(round(float(r["overround_wm"]), 4)
                                   if market[:4] in ("red_", "blue") and r["overround_wm"] == r["overround_wm"]
                                   else None)))
        db.add_all(rows)
        db.commit()
    finally:
        db.close()
    return len(rows)


def build(cache_dir: Path = DEFAULT_CACHE_DIR, odds_csv: Path = Path("data/bfo/odds.csv")) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(Path(cache_dir) / "pages" / "events_*.txt")):
        html = Path(f).read_text()
        if _json_ld_date(html) is None:
            continue
        rows.extend(parse_props(html))
    links = pd.read_csv(odds_csv, dtype={"db_fight_id": str},
                        usecols=["bfo_matchup_id", "db_fight_id", "db_swapped"])
    return to_corners(consensus(rows), links)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bfo/props_close.csv")
    ap.add_argument("--load-db", action="store_true",
                    help="also backfill the closes into ufc_prop_odds_history")
    a = ap.parse_args()
    df = build()
    df.to_csv(a.out, index=False)
    print(f"{len(df)} fights with prop closes -> {a.out}")
    print(df.notna().mean().round(2).to_string())
    if a.load_db:
        print(f"{backfill_closes(a.out)} rows loaded into ufc_prop_odds_history")


if __name__ == "__main__":
    main()
