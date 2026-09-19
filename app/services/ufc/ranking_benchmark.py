"""Agreement between a candidate ranking and the official UFC (Meta) rankings.

`ranking_eval` answers "does this ranking predict fight outcomes". That is necessary but
not sufficient: its own docstring concedes that at n~350 top-15 bouts it cannot adjudicate
ORDER quality, which is the thing a published ranking is actually judged on. Every scoring
change so far has come back statistically indistinguishable there, so a ranker can pass the
predictive gate while putting an undefeated prospect above a champion — which is exactly
what the first cut of the tiered ranker did.

This module supplies the missing half: an external reference. The UFC's own rankings have
been the Meta Elo model since 2026-06-20, so this is no longer a comparison against a media
panel's opinion — it is a comparison against another objective system built for the same
job, which makes disagreement informative rather than merely different.

Metrics, over the fighters present in both lists:

    spearman    rank correlation, the headline number
    mae         mean absolute rank difference
    top5_jacc   set overlap of the top 5 — where a ranking is actually read
    coverage    how many of the official top 15 the candidate ranks at all

Snapshot data, not a live fetch: a benchmark that changes under you cannot be used to
compare two rankers. Refresh with --show-stale and record the date.

    python -m app.services.ufc.ranking_benchmark
    python -m app.services.ufc.ranking_benchmark --rankers tiered,points
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

log = logging.getLogger("ranking_benchmark")

#: Official UFC rankings, captured 2026-09-18 from CBS Sports' mirror of ufc.com.
#: Champion first — the UFC lists them above the top 15 rather than at #1, but for rank
#: correlation they are position 1 and the numbered contenders follow.
SNAPSHOT_DATE = date(2026, 9, 18)

OFFICIAL: dict[str, list[str]] = {
    "heavyweight": [
        "Tom Aspinall", "Ciryl Gane", "Alexander Volkov", "Sergei Pavlovich",
        "Josh Hokit", "Curtis Blaydes", "Waldo Cortes Acosta", "Rizvan Kuniev",
        "Vitor Petrino", "Serghei Spivac", "Ante Delija", "Valter Walker",
        "Tyrell Fortune", "Derrick Lewis", "Mario Pinto", "Aleksandar Rakic",
    ],
    "lightweight": [
        "Justin Gaethje", "Ilia Topuria", "Arman Tsarukyan", "Charles Oliveira",
        "Max Holloway", "Paddy Pimblett", "Benoit Saint Denis", "Mauricio Ruffy",
        "Quillan Salkilld", "Salahdine Parnasse", "Rafael Fiziev", "Mateusz Gamrot",
        "Renato Moicano", "Dan Hooker", "Tom Nolan", "Beneil Dariush",
    ],
    "welterweight": [
        "Islam Makhachev", "Ian Machado Garry", "Carlos Prates", "Michael Morales",
        "Jack Della Maddalena", "Gabriel Bonfim", "Sean Brady", "Belal Muhammad",
        "Leon Edwards", "Kamaru Usman", "Joaquin Buckley", "Yaroslav Amosov",
        "Uros Medic", "Mike Malott", "Daniel Rodriguez", "Neil Magny",
    ],
    "middleweight": [
        "Sean Strickland", "Khamzat Chimaev", "Dricus Du Plessis", "Nassourdine Imavov",
        "Brendan Allen", "Caio Borralho", "Joe Pyfer", "Gregory Rodrigues",
        "Anthony Hernandez", "Israel Adesanya", "Christian Leroy Duncan",
        "Jared Cannonier", "Abus Magomedov", "Ikram Aliskerov", "Bo Nickal",
        "Shara Magomedov",
    ],
    "light_heavyweight": [
        "Carlos Ulberg", "Magomed Ankalaev", "Jiri Prochazka", "Alex Pereira",
        "Khalil Rountree Jr.", "Navajo Stirling", "Paulo Costa", "Jamahal Hill",
        "Azamat Murzakanov", "Jan Blachowicz", "Dominick Reyes", "Bogdan Guskov",
        "Robert Whittaker", "Johnny Walker", "Alonzo Menifield", "Nikita Krylov",
    ],
    "featherweight": [
        "Alexander Volkanovski", "Movsar Evloev", "Diego Lopes", "Lerone Murphy",
        "Aljamain Sterling", "Jean Silva", "Yair Rodriguez", "Arnold Allen",
        "Youssef Zalal", "Kevin Vallejos", "Steve Garcia", "Brian Ortega",
        "Aaron Pico", "Melquizael Costa", "David Onama", "Patricio Pitbull",
    ],
    "bantamweight": [
        "Petr Yan", "Merab Dvalishvili", "Sean O'Malley", "Song Yadong",
        "Umar Nurmagomedov", "Mario Bautista", "Cory Sandhagen", "Aiemann Zahabi",
        "David Martinez", "Deiveson Figueiredo", "Marlon Vera", "Payton Talbott",
        "Raul Rosas Jr.", "Raoni Barcelos", "Farid Basharat", "Marcus McGhee",
    ],
    "flyweight": [
        "Joshua Van", "Alexandre Pantoja", "Manel Kape", "Brandon Royval",
        "Tatsuro Taira", "Kyoji Horiguchi", "Lone'er Kavanagh", "Asu Almabayev",
        "Brandon Moreno", "Amir Albazi", "Ramazan Temirov", "Tim Elliott",
        "Steve Erceg", "Sumudaerji", "Tagir Ulanbekov", "Alex Perez",
    ],
}


def _spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    dbb = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * dbb) if da and dbb else float("nan")


def compare(candidate: list[str], official: list[str], top_n: int = 15) -> dict:
    """Agreement over the fighters both lists contain.

    Restricting to the intersection is the only fair comparison: the candidate ranks every
    eligible fighter in the division while the official list stops at 15, and a fighter the
    UFC has not ranked is not evidence that the candidate is wrong about them.
    """
    off = official[:top_n + 1]
    off_pos = {n: i for i, n in enumerate(off)}
    cand_pos = {n: i for i, n in enumerate(candidate)}
    shared = [n for n in off if n in cand_pos]
    if len(shared) < 2:
        return {"n": len(shared), "spearman": float("nan"), "mae": float("nan"),
                "top5_jaccard": float("nan"), "coverage": 0.0}

    # Re-rank within the shared set so a fighter the candidate cannot rank (retired,
    # inactive, never in this division) does not shift everyone below them.
    off_rank = {n: i for i, n in enumerate(sorted(shared, key=lambda x: off_pos[x]))}
    cand_rank = {n: i for i, n in enumerate(sorted(shared, key=lambda x: cand_pos[x]))}

    xs = [off_rank[n] for n in shared]
    ys = [cand_rank[n] for n in shared]
    top5_off = {n for n in off[:5] if n in cand_pos}
    top5_cand = set(candidate[:5])
    union = top5_off | top5_cand

    # Eye-test metric: of the official top 10, how many land in OUR top 10. This is the
    # question a reader actually asks ("why isn't Tsarukyan here?"), and it is not what
    # Spearman measures — Spearman is dominated by the long tail of the division, where
    # nobody is looking. Restricted to fighters we can rank at all, so a retired or
    # ineligible fighter is not counted as a miss.
    top10_off = [n for n in off[:10] if n in cand_pos]
    top10_cand = set(candidate[:10])
    top10_recall = (len([n for n in top10_off if n in top10_cand]) / len(top10_off)
                    if top10_off else float("nan"))

    return {
        "n": len(shared),
        "spearman": _spearman(xs, ys),
        "mae": sum(abs(x - y) for x, y in zip(xs, ys)) / len(shared),
        "top5_jaccard": len(top5_off & top5_cand) / len(union) if union else float("nan"),
        "top10_recall": top10_recall,
        "coverage": len(shared) / len(off),
    }


def run(ranker_names: list[str], as_of: date | None = None) -> dict:
    from app.database import SessionLocal
    from app.models.ufc import UFCFighter
    from app.services.ufc.fighter_registry import Eligibility, build_fighter_registry

    db = SessionLocal()
    try:
        today = as_of or date.today()
        registry = build_fighter_registry(db)
        names = {f.id: f"{f.first_name or ''} {f.last_name or ''}".strip()
                 for f in db.query(UFCFighter).all()}

        results: dict[str, dict] = {}
        for rn in ranker_names:
            if rn == "tiered" or rn.startswith("tiered@"):
                from app.services.ufc.tiered_ranking_service import TieredRanker
                c, mode = 1.0, "decay"
                if "@" in rn:
                    for part in rn.split("@", 1)[1].split(":"):
                        k, _, val = part.partition("=")
                        if k == "c":
                            c = float(val)
                        elif k == "mode":
                            mode = val
                ranker = TieredRanker(conservative=c, mode=mode)
            elif rn == "tapology":
                from app.services.ufc.tapology_ranking_service import TapologyRanker
                ranker = TapologyRanker()
            elif rn == "points":
                from app.services.ufc.points_ranking_service import PointsEloRanker
                ranker = PointsEloRanker()
            else:
                raise SystemExit(f"unknown ranker: {rn}")

            res = ranker.rank(db, registry, today, Eligibility())
            per_div = {}
            for div, official in OFFICIAL.items():
                cand = [names.get(f, "") for f in res.order.get(div, [])]
                per_div[div] = compare(cand, official)
            results[rn] = per_div
        return results
    finally:
        db.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankers", default="tiered,points")
    args = ap.parse_args()

    logging.disable(logging.INFO)
    names = [x.strip() for x in args.rankers.split(",") if x.strip()]
    results = run(names)

    print(f"\nAgreement with official UFC (Meta) rankings, snapshot {SNAPSHOT_DATE}")
    print(f"{'ranker':<10} {'division':<16} {'n':>3} {'spearman':>9} "
          f"{'mae':>6} {'top5':>6} {'top10':>6} {'cover':>6}")
    print("-" * 70)
    for rn, per_div in results.items():
        for div, m in per_div.items():
            print(f"{rn:<10} {div:<16} {m['n']:>3} {m['spearman']:>9.3f} "
                  f"{m['mae']:>6.2f} {m['top5_jaccard']:>6.2f} "
                  f"{m['top10_recall']:>6.2f} {m['coverage']:>6.2f}")
        vals = [m for m in per_div.values() if m["spearman"] == m["spearman"]]
        if vals:
            print(f"{rn:<10} {'MEAN':<16} {'':>3} "
                  f"{sum(v['spearman'] for v in vals)/len(vals):>9.3f} "
                  f"{sum(v['mae'] for v in vals)/len(vals):>6.2f} "
                  f"{sum(v['top5_jaccard'] for v in vals)/len(vals):>6.2f} "
                  f"{sum(v['top10_recall'] for v in vals)/len(vals):>6.2f} "
                  f"{sum(v['coverage'] for v in vals)/len(vals):>6.2f}")
        print("-" * 70)


if __name__ == "__main__":
    main()
