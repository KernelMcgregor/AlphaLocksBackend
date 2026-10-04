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
  sr_{N}_{yes|no}         "Fight starts round N" (lasts past the end of round N-1)
  er_{N}_{yes|no}         "Fight ends in round N"
  {msig|mtd|msigr1}_{a|b} "<fighter> has more significant strikes | takedowns |
                          significant strikes in round 1" (ties void; de-vigged as a pair)
  tdz_{a|b}_{yes|no}      "<fighter> lands no takedowns" / "lands at least one takedown"
Stat markets are thin: ~180-210 fights each, 2021+ (checked on the cache 2026-10-04).

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
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pandas is only needed for the offline CSV / backfill helpers; the line
    import pandas as pd  # watcher's light CI install does not have it

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
_MORE = re.compile(r" has more (significant strikes in round 1|significant strikes|takedowns)$")
_MORE_STAT = {"significant strikes": "msig", "takedowns": "mtd",
              "significant strikes in round 1": "msigr1"}


def _label_key(label: str, fighter: int | None) -> str | None:
    m = _WM.search(label)
    if m and fighter in (1, 2):
        return f"wm_{'ab'[fighter - 1]}_{_METHOD[m.group(1)]}"
    m = re.search(r"wins inside distance$", label)
    if m and fighter in (1, 2):
        return f"itd_{'ab'[fighter - 1]}_yes"
    if label.startswith("Not ") and label.endswith(" inside distance") and fighter in (1, 2):
        return f"itd_{'ab'[fighter - 1]}_no"
    if label.startswith("Not ") and label.endswith(" by decision") and fighter in (1, 2):
        return f"wm_{'ab'[fighter - 1]}_dec_no"
    if label == "Fight goes to decision":
        return "dec_yes"
    if label == "Fight doesn't go to decision":
        return "dec_no"
    m = re.match(r"^Fight (starts|won't start) round (\d)$", label)
    if m:
        return f"sr_{m.group(2)}_{'yes' if m.group(1) == 'starts' else 'no'}"
    m = re.match(r"^Fight (ends|doesn't end) in round (\d)$", label)
    if m:
        return f"er_{m.group(2)}_{'yes' if m.group(1) == 'ends' else 'no'}"
    m = _OU.match(label)
    if m:
        line = int(m.group(2)) + (0.5 if m.group(3) else 0.0)
        return f"ou_{line}_{m.group(1).lower()}"
    m = _MORE.search(label)
    if m and fighter in (1, 2):
        return f"{_MORE_STAT[m.group(1)]}_{'ab'[fighter - 1]}"
    if fighter in (1, 2) and label.endswith(" lands no takedowns"):
        return f"tdz_{'ab'[fighter - 1]}_yes"
    if fighter in (1, 2) and label.endswith(" lands at least one takedown"):
        return f"tdz_{'ab'[fighter - 1]}_no"
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


def _american(dec: float) -> int:
    return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1)))


def consensus(rows: list[dict], book_names: dict[int, str] | None = None) -> dict[int, dict]:
    """matchup -> {key: de-vigged consensus prob, 'n_books', 'overround_wm', ...,
    'prices': {raw key: (best American, best book, median American)}}."""
    by = defaultdict(lambda: defaultdict(list))
    books = defaultdict(set)
    for r in rows:
        p = american_to_prob(r["american"])
        if p is not None:
            by[r["matchup"]][r["key"]].append((p, 1 / p, r["book"]))
            books[r["matchup"]].add(r["book"])
    out = {}
    for mu, keys in by.items():
        med = {k: float(np.median([x[0] for x in v])) for k, v in keys.items()}
        prices = {}
        for k, v in keys.items():
            best = max(v, key=lambda x: x[1])
            name = (book_names or {}).get(best[2], f"book_{best[2]}")
            prices[k] = (_american(best[1]), name, _american(float(np.median([x[1] for x in v]))))
        res = {"n_books": len(books[mu]), "prices": prices}
        wm = [f"wm_{s}_{m}" for s in "ab" for m in ("ko", "sub", "dec")]
        if all(k in med for k in wm):
            res["overround_wm"] = sum(med[k] for k in wm)
        res.update(_devig_group(med, wm))
        res.update({f"{k}_nv": v for k, v in _devig_group(med, ["dec_yes", "dec_no"]).items()})
        if "dec_yes" in med and "dec_yes_nv" not in res:
            res["dec_yes_raw"] = med["dec_yes"]
        for k in med:
            for fam in ("sr_", "itd_a_", "itd_b_"):  # yes/no pairs
                if k.startswith(fam) and k.endswith("_yes"):
                    n = k[len(fam):-4]
                    res.update({f"{kk}_nv": v for kk, v in
                                _devig_group(med, [f"{fam}{n}_yes", f"{fam}{n}_no"]).items()})
            if k.startswith("tdz_") and k.endswith("_yes"):
                fam = k[:-4]
                res.update({f"{kk}_nv": v for kk, v in
                            _devig_group(med, [f"{fam}_yes", f"{fam}_no"]).items()})
            if k in ("msig_a", "mtd_a", "msigr1_a"):   # "A has more X" vs "B has more X"
                stat = k[:-2]
                res.update({f"{kk}_nv": v for kk, v in
                            _devig_group(med, [f"{stat}_a", f"{stat}_b"]).items()})
            if k.startswith("ou_") and k.endswith("_over"):
                line = k[3:-5]
                res.update({f"{kk}_nv": v for kk, v in
                            _devig_group(med, [f"ou_{line}_over", f"ou_{line}_under"]).items()})
        # "Ends in round N": books price only the Yes side, but rounds 1..K plus "goes to
        # decision" partition every outcome, so de-vig them together (er_N_part).
        rounds = sorted(int(k[3]) for k in med if k.startswith("er_") and k.endswith("_yes"))
        if rounds and rounds == list(range(1, rounds[-1] + 1)) and rounds[-1] in (3, 5) \
                and "dec_yes" in med:
            keys_ = [f"er_{n}_yes" for n in rounds] + ["dec_yes"]
            tot = sum(med[k] for k in keys_)
            res.update({f"er_{n}_part": med[f"er_{n}_yes"] / tot for n in rounds})
            res["overround_er"] = tot
        out[mu] = res
    return out


def corner_markets(c: dict, swapped: bool) -> dict[str, dict]:
    """One matchup's consensus -> {market: {prob, overround, best_american, best_book,
    median_american}} in red/blue terms. Both sides of two-way markets are listed
    (dec_yes / dec_no, ou_X_over / ou_X_under, sr_N / sr_N_no, itd_<corner>_yes / _no).
    swapped: BFO fighter_a is the DB blue corner (bfo_scraper.match_matchups)."""
    a, b = ("blue", "red") if swapped else ("red", "blue")
    corner = {"a": a, "b": b}
    px = c.get("prices", {})
    out = {}

    def put(market, prob, raw_key, overround=None):
        if prob is None:
            return
        best, book, median = px.get(raw_key, (None, None, None))
        out[market] = {"prob": prob, "overround": overround, "best_american": best,
                       "best_book": book, "median_american": median}

    for m in ("ko", "sub", "dec"):
        for side in "ab":
            put(f"{corner[side]}_{m}", c.get(f"wm_{side}_{m}"), f"wm_{side}_{m}", c.get("overround_wm"))
    for side in "ab":
        yes = c.get(f"itd_{side}_yes_nv")
        if yes is None and c.get(f"wm_{side}_ko") is not None and c.get(f"wm_{side}_sub") is not None:
            yes = c[f"wm_{side}_ko"] + c[f"wm_{side}_sub"]   # from the de-vigged 6-way grid
        if yes is not None:
            put(f"itd_{corner[side]}_yes", yes, f"itd_{side}_yes")
            put(f"itd_{corner[side]}_no", 1 - yes, f"itd_{side}_no")
    put("dec_yes", c.get("dec_yes_nv"), "dec_yes")
    put("dec_no", c.get("dec_no_nv"), "dec_no")
    for k, v in c.items():
        if k.startswith("ou_") and k.endswith("_nv"):
            put(k[:-3], v, k[:-3])                       # ou_1.5_over / ou_1.5_under
        elif k.startswith("sr_") and k.endswith("_yes_nv"):
            put(k[:-7], v, k[:-3])                       # sr_2 = P(fight starts round 2)
        elif k.startswith("sr_") and k.endswith("_no_nv"):
            put(k[:-3], v, k[:-3])                       # sr_2_no
        elif k.startswith("er_") and k.endswith("_part"):
            put(k[:-5], v, f"{k[:-5]}_yes", c.get("overround_er"))   # er_1 = ends in round 1
    for stat in ("msig", "mtd", "msigr1"):   # more_sig_red = P(red lands more), ties void
        for side in "ab":
            put(f"more_{stat[1:]}_{corner[side]}", c.get(f"{stat}_{side}_nv"), f"{stat}_{side}")
    for side in "ab":                        # tdz_red = P(red lands no takedowns)
        put(f"tdz_{corner[side]}", c.get(f"tdz_{side}_yes_nv"), f"tdz_{side}_yes")
        put(f"tdz_{corner[side]}_no", c.get(f"tdz_{side}_no_nv"), f"tdz_{side}_no")
    return out


def to_corners(cons: dict[int, dict], links: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    """links: bfo_matchup_id, db_fight_id, db_swapped (from data/bfo/odds.csv)."""
    link = links.dropna(subset=["db_fight_id"]).drop_duplicates("bfo_matchup_id")
    rows = []
    for mu, fid, swapped in zip(link["bfo_matchup_id"], link["db_fight_id"], link["db_swapped"]):
        c = cons.get(int(mu))
        if not c:
            continue
        r = {"fight_id": str(fid).split(".")[0], "bfo_matchup_id": int(mu), "n_books": c["n_books"],
             "overround_wm": c.get("overround_wm")}
        r.update({f"mkt_{k}": q["prob"] for k, q in corner_markets(c, str(swapped) == "True").items()})
        rows.append(r)
    return pd.DataFrame(rows)


def parse_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[int, dict]:
    """Consensus (with best/median prices and book names) for every cached event page."""
    from app.services.ufc.bfo_scraper import parse_event_page
    cons = {}
    for f in glob.glob(str(Path(cache_dir) / "pages" / "events_*.txt")):
        html = Path(f).read_text()
        if _json_ld_date(html) is None:
            continue
        try:
            names = parse_event_page(html).books
        except ValueError:
            names = {}
        cons.update(consensus(parse_props(html), names))
    return cons


def backfill_closes(odds_csv: Path = Path("data/bfo/odds.csv"), replace: bool = False) -> int:
    """Load closing props (de-vigged prob + best/median price and book) from the cached event
    pages into ufc_prop_odds_history (source 'bfo_close', captured_at = event date).
    Fights that already have bfo_close rows are skipped unless replace=True."""
    import pandas as pd
    import datetime as _dt

    from app.database import SessionLocal
    from app.models.ufc import UFCEvent, UFCFight, UFCPropOddsHistory

    cons = parse_cache()
    links = (pd.read_csv(odds_csv, dtype={"db_fight_id": str},
                         usecols=["bfo_matchup_id", "db_fight_id", "db_swapped"])
             .dropna(subset=["db_fight_id"]).drop_duplicates("bfo_matchup_id"))
    db = SessionLocal()
    try:
        dates = dict(db.query(UFCFight.id, UFCEvent.date).join(UFCEvent, UFCEvent.id == UFCFight.event_id))
        if replace:
            db.query(UFCPropOddsHistory).filter(UFCPropOddsHistory.source == "bfo_close").delete()
            db.commit()
        have = {f for (f,) in db.query(UFCPropOddsHistory.fight_id)
                .filter(UFCPropOddsHistory.source == "bfo_close").distinct()}
        rows = []
        for mu, fid, swapped in zip(links["bfo_matchup_id"], links["db_fight_id"], links["db_swapped"]):
            fid = int(str(fid).split(".")[0])
            c = cons.get(int(mu))
            if not c or fid in have or dates.get(fid) is None:
                continue
            ts = _dt.datetime.combine(dates[fid], _dt.time())
            for market, q in corner_markets(c, str(swapped) == "True").items():
                rows.append(UFCPropOddsHistory(
                    fight_id=fid, market=market, prob=round(float(q["prob"]), 4),
                    n_books=c["n_books"], source="bfo_close", captured_at=ts,
                    overround=round(q["overround"], 4) if q["overround"] else None,
                    best_american=q["best_american"], best_book=q["best_book"],
                    median_american=q["median_american"]))
            have.add(fid)
        db.add_all(rows)
        db.commit()
    finally:
        db.close()
    return len(rows)


def build(cache_dir: Path = DEFAULT_CACHE_DIR, odds_csv: Path = Path("data/bfo/odds.csv")) -> pd.DataFrame:
    import pandas as pd
    links = pd.read_csv(odds_csv, dtype={"db_fight_id": str},
                        usecols=["bfo_matchup_id", "db_fight_id", "db_swapped"])
    return to_corners(parse_cache(cache_dir), links)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bfo/props_close.csv")
    ap.add_argument("--load-db", action="store_true",
                    help="also backfill the closes into ufc_prop_odds_history")
    ap.add_argument("--replace", action="store_true",
                    help="with --load-db: replace existing bfo_close rows (e.g. to add prices)")
    a = ap.parse_args()
    df = build()
    df.to_csv(a.out, index=False)
    print(f"{len(df)} fights with prop closes -> {a.out}")
    print(df.notna().mean().round(2).to_string())
    if a.load_db:
        print(f"{backfill_closes(replace=a.replace)} rows loaded into ufc_prop_odds_history")


if __name__ == "__main__":
    main()
