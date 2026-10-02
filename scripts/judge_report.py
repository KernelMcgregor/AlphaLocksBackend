"""Per-judge report from ufc_judge_scorecards (mmadecisions round cards).

For every judge with enough verified UFC decisions:
  fights / rounds          sample size
  minority_rate            share of fights where the judge's winner differs from the
                           other two judges, when those two agree
  round_dissent            same, per round (pre-deduction scores)
  r108_rate, r1010_rate    share of rounds scored 10-8 (or wider) / 10-10
  vs_media                 share of fights where the judge's winner matches the media
                           majority (>= 3 media scores)
  vs_fans                  share of rounds matching the fans' majority pick
  vs_model                 share of rounds matching the round model (scripts.round_model)
  ctrl_lean                in rounds where the control leader and the sig-strike leader
                           differ: share the judge gave to the control leader, minus the
                           share the other judges on the same rounds gave (+ = favours
                           control/wrestling, - = favours striking)

    DATABASE_URL=postgresql://localhost/alocks_local PYTHONPATH=. \
        venv/bin/python -m scripts.judge_report --min-fights 30 --out data/mmad/judges.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-fights", type=int, default=30)
    ap.add_argument("--round-probs", type=Path, default=Path("data/mmad/round_probs.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/mmad/judges.csv"))
    args = ap.parse_args(argv)
    eng = create_engine(os.environ["DATABASE_URL"])

    cards = pd.read_sql(text("""
        SELECT c.mmad_decision_id, c.fight_id, c.round, c.judge_id, j.name AS judge,
               c.red_pts + c.red_ded AS red, c.blue_pts + c.blue_ded AS blue
        FROM ufc.ufc_judge_scorecards c
        JOIN ufc.ufc_judges j ON j.id = c.judge_id
        JOIN ufc.mmad_decisions d ON d.mmad_decision_id = c.mmad_decision_id
        WHERE c.source = 'mmad' AND d.match_status = 'verified'
          AND c.red_pts IS NOT NULL AND c.blue_pts IS NOT NULL
    """), eng)
    cards["s"] = np.sign(cards.red - cards.blue)            # +1 red, -1 blue, 0 even

    def others(df: pd.DataFrame) -> pd.DataFrame:
        """Attach the other judges' verdicts on the same fight-round."""
        g = df.groupby(["fight_id", "round"])["s"]
        df = df.assign(sum_s=g.transform("sum"), n=g.transform("size"))
        df["oth_n"] = df.n - 1
        df["oth_sum"] = df.sum_s - df.s
        # Other two agree (both +1 or both -1) and the judge differs.
        df["oth_agree"] = (df.oth_n == 2) & (df.oth_sum.abs() == 2)
        df["dissent"] = df.oth_agree & (df.s != np.sign(df.oth_sum))
        return df

    tot = others(cards[cards["round"] == 0].copy())
    rnd = others(cards[cards["round"] >= 1].copy())
    rnd["margin"] = (rnd.red - rnd.blue).abs()

    # Media majority per fight (in DB corners).
    media = pd.read_sql(text("""
        SELECT d.fight_id, d.swapped, m.pick FROM ufc.mmad_media_scores m
        JOIN ufc.mmad_decisions d ON d.mmad_decision_id = m.mmad_decision_id
        WHERE d.match_status = 'verified' AND m.pick IN ('a', 'b')
    """), eng)
    media["s"] = np.where((media.pick == "a") != media.swapped.astype(bool), 1, -1)
    mm = media.groupby("fight_id").s.agg(["sum", "size"])
    mm = mm[(mm["size"] >= 3) & (mm["sum"] != 0)]
    tot = tot.join(np.sign(mm["sum"]).rename("media_s"), on="fight_id")

    fans = pd.read_sql(text("""
        SELECT d.fight_id, d.swapped, f.round, f.pick, f.pct FROM ufc.mmad_fan_scores f
        JOIN ufc.mmad_decisions d ON d.mmad_decision_id = f.mmad_decision_id
        WHERE d.match_status = 'verified' AND f.round >= 1
    """), eng)
    top = fans.sort_values("pct").groupby(["fight_id", "round"]).tail(1)
    top = top[top.pick.isin(["a", "b"])]
    top["fan_s"] = np.where((top.pick == "a") != top.swapped.astype(bool), 1, -1)
    rnd = rnd.merge(top[["fight_id", "round", "fan_s"]], on=["fight_id", "round"], how="left")

    if args.round_probs.exists():
        rp = pd.read_csv(args.round_probs)[["fight_id", "round", "p_red"]]
        rnd = rnd.merge(rp, on=["fight_id", "round"], how="left")
        rnd["model_s"] = np.sign(rnd.p_red - 0.5)
    else:
        rnd["model_s"] = np.nan

    stats = pd.read_sql(text("""
        SELECT s.fight_id, s.round_number AS round,
               SUM(CASE WHEN s.fighter_id = f.red_fighter_id THEN s.ctrl_seconds ELSE -s.ctrl_seconds END) AS d_ctrl,
               SUM(CASE WHEN s.fighter_id = f.red_fighter_id THEN s.sig_str_landed ELSE -s.sig_str_landed END) AS d_sig
        FROM ufc.ufc_fight_stats s JOIN ufc.ufc_fights f ON f.id = s.fight_id
        WHERE s.round_number >= 1 GROUP BY s.fight_id, s.round_number
    """), eng)
    rnd = rnd.merge(stats, on=["fight_id", "round"], how="left")
    rnd["ctrl_leader"] = np.sign(rnd.d_ctrl)
    conflict = (rnd.ctrl_leader != 0) & (np.sign(rnd.d_sig) != 0) & \
               (rnd.ctrl_leader != np.sign(rnd.d_sig)) & (rnd.s != 0)
    rnd["to_ctrl"] = np.where(conflict, (rnd.s == rnd.ctrl_leader).astype(float), np.nan)
    # Panel share on the same rounds, excluding this judge.
    g = rnd[conflict].groupby(["fight_id", "round"])["to_ctrl"]
    rnd.loc[conflict, "oth_ctrl"] = (g.transform("sum") - rnd.loc[conflict, "to_ctrl"]) / \
        (g.transform("size") - 1).replace(0, np.nan)

    rows = []
    for (jid, name), t in tot.groupby(["judge_id", "judge"]):
        r = rnd[rnd.judge_id == jid]
        if t.fight_id.nunique() < args.min_fights:
            continue
        ag, rag = t[t.oth_agree], r[r.oth_agree]
        md, fn, mo = t[t.media_s.notna()], r[r.fan_s.notna()], r[r.model_s.notna() & (r.s != 0)]
        c = r[r.to_ctrl.notna() & r.oth_ctrl.notna()]
        rows.append(dict(
            judge_id=jid, judge=name, fights=t.fight_id.nunique(), rounds=len(r),
            first=None, minority_rate=ag.dissent.mean() if len(ag) else np.nan,
            round_dissent=rag.dissent.mean() if len(rag) else np.nan,
            r108_rate=(r.margin >= 2).mean(), r1010_rate=(r.s == 0).mean(),
            vs_media=(md.s == md.media_s).mean() if len(md) else np.nan,
            vs_fans=(fn.s == fn.fan_s).mean() if len(fn) else np.nan,
            vs_model=(mo.s == mo.model_s).mean() if len(mo) else np.nan,
            ctrl_lean=(c.to_ctrl - c.oth_ctrl).mean() if len(c) >= 30 else np.nan,
            ctrl_rounds=len(c),
        ))
    out = pd.DataFrame(rows).drop(columns="first").sort_values("fights", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).head(40).to_string(index=False))
    print(f"\n{len(out)} judges with >= {args.min_fights} fights -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
