"""Round-scoring model: P(judges give the round to red | that round's UFCStats stats).

Labels come from mmadecisions round cards (ufc_judge_scorecards, source='mmad', verified
fights), taken BEFORE referee deductions (pts + ded), so a foul does not turn a judge's
10-9 into a 10-10. 10-10 rounds are dropped. One row per judge-round.

Features are red-minus-blue differences of the per-round stats in ufc_fight_stats.
Trained on fights before --split, evaluated on fights from --split on (a time split, so
whole events are held out). Reports log loss and accuracy against each judge and the
panel majority, the fitted weights ("what wins rounds"), the same fit before and after
the 2017 scoring-criteria change, and writes P(red won the round) for EVERY complete
round of every fight with round stats, finished fights included (--out).

    DATABASE_URL=postgresql://localhost/alocks_local PYTHONPATH=. \
        venv/bin/python -m scripts.round_model --out data/mmad/round_probs.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sqlalchemy import create_engine, text

STATS = ["kd", "sig_str_landed", "sig_str_attempted", "total_str_landed", "head_landed",
         "body_landed", "leg_landed", "ground_landed", "td_landed", "td_attempted",
         "sub_att", "rev", "ctrl_seconds"]
RULES_2017 = dt.date(2017, 1, 1)


def load_round_stats(eng) -> pd.DataFrame:
    """One row per fight-round with red-minus-blue stat differences."""
    cols = ", ".join(f"s.{c}" for c in STATS)
    df = pd.read_sql(text(f"""
        SELECT s.fight_id, s.round_number AS round, s.fighter_id, {cols},
               f.red_fighter_id, f.blue_fighter_id, f.method, f.finish_round,
               COALESCE(f.date, e.date) AS date
        FROM ufc.ufc_fight_stats s
        JOIN ufc.ufc_fights f ON f.id = s.fight_id
        JOIN ufc.ufc_events e ON e.id = f.event_id
        WHERE s.round_number >= 1
    """), eng)
    df[STATS] = df[STATS].astype(float).fillna(0.0)
    red = df[df.fighter_id == df.red_fighter_id].set_index(["fight_id", "round"])
    blue = df[df.fighter_id == df.blue_fighter_id].set_index(["fight_id", "round"])
    both = red.index.intersection(blue.index)
    out = (red.loc[both, STATS] - blue.loc[both, STATS]).add_prefix("d_")
    meta = red.loc[both, ["date", "method", "finish_round", "red_fighter_id", "blue_fighter_id"]]
    out = out.join(meta).reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    # A finish ends the round it happens in: only rounds before it were complete.
    dec = out["method"].fillna("").str.startswith("Decision")
    out["complete"] = dec | (out["round"] < out["finish_round"].fillna(99))
    return out


def load_cards(eng) -> pd.DataFrame:
    return pd.read_sql(text("""
        SELECT c.fight_id, c.round, c.judge_id, c.judge_seq,
               c.red_pts + c.red_ded AS red, c.blue_pts + c.blue_ded AS blue,
               c.red_pts, c.blue_pts
        FROM ufc.ufc_judge_scorecards c
        JOIN ufc.mmad_decisions d ON d.mmad_decision_id = c.mmad_decision_id
        WHERE c.source = 'mmad' AND c.round >= 1 AND d.match_status = 'verified'
          AND c.red_pts IS NOT NULL AND c.blue_pts IS NOT NULL
    """), eng)


def _features(df: pd.DataFrame) -> np.ndarray:
    return df[[f"d_{c}" for c in STATS]].to_numpy()


def fit(df: pd.DataFrame) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    X = _features(df)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, df["y"])
    return m, mu, sd


def predict(m, mu, sd, df: pd.DataFrame) -> np.ndarray:
    return m.predict_proba((_features(df) - mu) / sd)[:, 1]


def weights(m, sd) -> pd.Series:
    # Per raw unit of each stat difference (e.g. per extra sig strike landed).
    return pd.Series(m.coef_[0] / sd, index=STATS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=dt.date.fromisoformat, default=dt.date(2023, 1, 1))
    ap.add_argument("--out", type=Path, default=Path("data/mmad/round_probs.csv"))
    args = ap.parse_args(argv)
    eng = create_engine(os.environ["DATABASE_URL"])

    rounds = load_round_stats(eng)
    cards = load_cards(eng)
    cards = cards[cards.red != cards.blue].copy()
    cards["y"] = (cards.red > cards.blue).astype(int)
    data = cards.merge(rounds, on=["fight_id", "round"], how="inner")
    print(f"judge-rounds with stats: {len(data):,} ({data.fight_id.nunique():,} fights); "
          f"10-10s dropped; {(cards.red_pts == cards.blue_pts).sum():,} posted level "
          f"of which {((cards.red_pts == cards.blue_pts) & (cards.red != cards.blue)).sum():,} "
          f"were deduction rounds")

    tr, te = data[data.date < args.split], data[data.date >= args.split]
    m, mu, sd = fit(tr)
    p = predict(m, mu, sd, te)
    base = np.clip(0.5 + 0.5 * np.sign(te["d_sig_str_landed"]), 0.02, 0.98)
    print(f"\nheld out ({args.split}+): {len(te):,} judge-rounds")
    print(f"  log loss  model {log_loss(te.y, p):.4f}   sig-strike sign {log_loss(te.y, base):.4f}"
          f"   coin {log_loss(te.y, np.full(len(te), 0.5)):.4f}")
    print(f"  accuracy vs each judge  {((p > 0.5) == te.y).mean():.3f}")
    maj = te.groupby(["fight_id", "round"]).agg(y=("y", "mean"), n=("y", "size"))
    maj = maj[(maj.n >= 2) & (maj.y != 0.5)]
    pm = te.groupby(["fight_id", "round"]).apply(lambda g: predict(m, mu, sd, g.iloc[:1])[0],
                                                 include_groups=False)
    agree = ((pm.loc[maj.index] > 0.5) == (maj.y > 0.5)).mean()
    unanimous = maj[(maj.y == 0) | (maj.y == 1)]
    print(f"  accuracy vs panel majority {agree:.3f}   "
          f"(unanimous rounds {((pm.loc[unanimous.index] > 0.5) == (unanimous.y > 0.5)).mean():.3f},"
          f" split rounds {((pm.loc[maj.index.difference(unanimous.index)] > 0.5) == (maj.drop(unanimous.index).y > 0.5)).mean():.3f})")

    full, fmu, fsd = fit(data)
    w = weights(full, fsd)
    print("\nwhat wins rounds (log-odds per unit of red-minus-blue difference, all data):")
    for k, v in w.sort_values(key=abs, ascending=False).items():
        print(f"  {k:<18} {v:+.4f}")
    pre, post = data[data.date < RULES_2017], data[data.date >= RULES_2017]
    if len(pre) > 1000 and len(post) > 1000:
        a, _, asd = fit(pre)
        b, _, bsd = fit(post)
        print("\nbefore vs after 2017 criteria change (per unit):")
        for k in STATS:
            print(f"  {k:<18} {weights(a, asd)[k]:+.4f}  {weights(b, bsd)[k]:+.4f}")

    comp = rounds[rounds.complete].copy()
    comp["p_red"] = predict(full, fmu, fsd, comp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    comp[["fight_id", "round", "date", "p_red"]].to_csv(args.out, index=False)
    print(f"\nwrote {len(comp):,} complete rounds ({comp.fight_id.nunique():,} fights) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
