"""Hypothetical bouts: the served models run on any two fighters, on request.

Every served prediction is made by the nightly batch (model.generate_predictions and the
method / rounds chain after it), which only knows booked bouts. This module scores pairs
that will never be booked — "what if A fought B at welterweight" — with the same three
models: the winner ensemble, method_v2 and rounds_v1.

How a fighter's features are built
----------------------------------
Each fighter gets an "as of today" feature row by the exact path a booked bout takes:
one unplayed PROBE bout per fighter, dated today, is appended through
`load_fight_data(extra_unplayed=...)` and goes through `build_features` with the rest of
history. Every rolling window, streak, Elo and career count on that row therefore
includes the fighter's latest fight — which copying their last historical row would not
(that row is pre-fight, so it omits the most recent result). Glicko gets the same probes
injected into the in-memory run, which records each fighter's rating AFTER their latest
bout (the unplayed-bout branch of glicko_service._run_glicko).

Probes are paired arbitrarily (consecutive fighters), so the few per-fighter columns that
depend on WHO the opponent is or WHICH division the bout is at are recomputed per request:

  opponent    elo_expected (from both Elo ratings), style_matchup_adv (style-vs-style
              rate after every played fight)
  division    division_rank, division_rank_delta, moved_up, moved_down, changed_division,
              fights_in_current_division, is_catchweight; the fight-level div_* one-hot,
              is_title_fight and the round count
  unknown     xs_* expected stats are fit per matchup inside build_features and are left
              NaN for the train-means imputation, exactly as serving does for a bout it
              has no matchup row for. bio_is_home is NaN (no venue).

The probe build is the full feature pipeline (15-20 minutes against the remote DB), so it
is cached (in memory and on disk) and
rebuilt once a day; a request against a cold cache gets `NotReady` while it builds.

No odds exist for a bout that will not happen, so the probability is the model's own
(`model_prob`), never the market blend. The pair is scored in both corner orientations
and averaged, so swapping the corners mirrors the answer exactly.

Nothing here writes to the database.
"""
from __future__ import annotations

import logging
import pickle
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd

log = logging.getLogger("hypothetical")

STATE_FILE = "hypothetical_state.pkl"
#: Rebuilt after this long. A day matches the nightly prediction job.
MAX_AGE = timedelta(hours=24)
#: Probe fight ids: negative, far from any real id.
PROBE_ID_BASE = -1_000_000_000

# Division keys the page can ask for, lightest first within each ladder, and the bout name
# each one is written as — the string every classifier downstream parses.
DIVISIONS = {
    "flyweight": "Flyweight", "bantamweight": "Bantamweight", "featherweight": "Featherweight",
    "lightweight": "Lightweight", "welterweight": "Welterweight", "middleweight": "Middleweight",
    "light_heavyweight": "Light Heavyweight", "heavyweight": "Heavyweight",
    "w_strawweight": "Women's Strawweight", "w_flyweight": "Women's Flyweight",
    "w_bantamweight": "Women's Bantamweight", "w_featherweight": "Women's Featherweight",
    "catchweight": "Catch Weight",
}

# Per-fighter columns recomputed per request (see the module docstring).
DIVISION_COLS = ("division_rank", "division_rank_delta", "moved_up", "moved_down",
                 "changed_division", "fights_in_current_division", "is_catchweight")


class NotReady(Exception):
    """The feature cache is being built; try again shortly."""


class UnknownFighter(Exception):
    pass


def bout_name(division: str, title: bool = False) -> str:
    """'light_heavyweight', title -> 'UFC Light Heavyweight Title Bout'."""
    label = DIVISIONS.get(division)
    if label is None:
        raise ValueError(f"unknown division {division!r}")
    return f"UFC {label} Title Bout" if title else f"{label} Bout"


def is_womens(division: str | None) -> bool:
    return bool(division) and division.startswith("w_")


# ---------------------------------------------------------------------------
# The cached state
# ---------------------------------------------------------------------------

@dataclass
class State:
    built_at: datetime
    as_of: date
    #: index fighter_id (int). Winner-model per-fighter columns plus `fighter_style`,
    #: `last_division` and `n_ufc_fights`.
    rows: pd.DataFrame
    #: index fighter_id. Method ratings and the rounds model's per-fighter history.
    per: pd.DataFrame
    fighter_cols: list[str]
    raw_cols: list[str]
    fight_cols: list[str]
    #: style_adv[i, j] = win rate of style i against style j after every played fight.
    style_adv: np.ndarray
    meta: dict = field(default_factory=dict)


def _probe_date(df_dates: pd.Series) -> date:
    last = max((d for d in df_dates if d), default=None)
    today = date.today()
    return today if last is None or today > last else last + timedelta(days=1)


def _style_table(feat: pd.DataFrame, n_styles: int) -> np.ndarray:
    """Style-vs-style rate after every played bout, read off the same expanding loop
    build_features uses: one synthetic bout per style pair, dated after everything, so
    each reads the matrix as it stands at the end of history."""
    from app.services.ufc.model import compute_style_matchup_features

    played = feat[feat["fight_id"] > 0]
    style_at = {(f, p): int(s) for f, p, s in
                zip(played["fight_id"], played["stats_fighter_id"], played["fighter_style"])}
    fights = played[["fight_id", "date", "red_fighter_id", "blue_fighter_id", "winner_id"]]
    end = pd.Timestamp.max.date()
    extra = []
    for i in range(n_styles):
        for j in range(n_styles):
            fid = PROBE_ID_BASE * 2 - (i * n_styles + j)
            a, b = -(10 + i), -(100 + j)            # stand-in fighters carrying each style
            style_at[(fid, a)], style_at[(fid, b)] = i, j
            extra.append({"fight_id": fid, "date": end, "red_fighter_id": a,
                          "blue_fighter_id": b, "winner_id": None})
    adv = compute_style_matchup_features(
        pd.concat([fights, pd.DataFrame(extra)], ignore_index=True), style_at, n_styles)
    table = np.full((n_styles, n_styles), 0.5)
    for i in range(n_styles):
        for j in range(n_styles):
            table[i, j] = adv.get((PROBE_ID_BASE * 2 - (i * n_styles + j), -(10 + i)), 0.5)
    return table


def build_state() -> State:
    """Feature rows for every fighter as of today. Slow (the full feature pipeline)."""
    from sqlalchemy import func, or_

    from app.database import SessionLocal
    from app.models.ufc import UFCFight, UFCFighter
    from app.services.ufc import glicko_service, method_ratings
    from app.services.ufc.model import (
        WINNER_RAW_COLS, _classify_weight_class, _is_fight_level, build_features,
        load_fight_data, winner_feature_columns,
    )
    from app.services.ufc.rounds_v1 import _per_fighter

    t0 = time.time()
    # Each fighter's last division and bout count, to place the probes after all history
    # and in the right division. A light query: the full load happens once, below.
    db = SessionLocal()
    try:
        fighter_ids = [int(f) for (f,) in db.query(UFCFighter.id)]
        bouts = (
            db.query(UFCFight.red_fighter_id, UFCFight.blue_fighter_id, UFCFight.weight_class,
                     UFCFight.date)
            .filter(or_(UFCFight.winner_id.isnot(None), func.coalesce(UFCFight.method, "") != ""))
            .order_by(UFCFight.date, UFCFight.id)
            .all()
        )
    finally:
        db.close()
    last_wc: dict[int, str] = {}
    n_fights: dict[int, int] = {}
    # Bouts in a row at the fighter's latest division, ending with the latest — what
    # build_features' fights_in_current_division reads for their next bout there. Taken
    # from here, not the probe row: a probe sits in its RED fighter's division, so the
    # blue fighter's count on it is measured against the wrong division.
    div_run: dict[int, int] = {}
    for r, b, wc, _d in bouts:
        div = _classify_weight_class(wc)
        for fid in (int(r), int(b)):
            same = fid in last_wc and _classify_weight_class(last_wc[fid]) == div
            div_run[fid] = div_run.get(fid, 0) + 1 if same else 1
            last_wc[fid] = wc
            n_fights[fid] = n_fights.get(fid, 0) + 1
    when = _probe_date(pd.Series([d for *_, d in bouts]))

    # Pair consecutive fighters. With an odd count, leave out one fighter who has never
    # fought in the UFC — nobody can be predicted without a row, and that is the least
    # interesting one to lose.
    ids = sorted(fighter_ids, key=lambda f: (-n_fights.get(f, 0), f))
    if len(ids) % 2:
        ids.pop()  # sorted by fight count, so the last one has the fewest
    probes = []
    for k in range(0, len(ids), 2):
        a, b = ids[k], ids[k + 1]
        probes.append(SimpleNamespace(
            id=PROBE_ID_BASE - k // 2, date=when, event_id=None, card_position=None,
            red_fighter_id=a, blue_fighter_id=b,
            # Each probe sits in its red fighter's last division; the division columns are
            # recomputed per request anyway, from `last_division` below.
            weight_class=last_wc.get(a) or "Lightweight Bout",
            time_format="5-5-5", event=None, scheduled_format=None,
        ))
    log.info(f"  {len(probes)} probe bouts dated {when} for {len(ids)} fighters")

    df, rd = load_fight_data(extra_unplayed=probes)

    # Glicko, in memory, with the probes injected as unplayed bouts so each fighter's
    # snapshot is their rating AFTER their latest fight. Same in-memory path training uses.
    fight_map, rounds_by_fight, _, fighter_info, baselines = glicko_service.load_glicko_data()
    for p in probes:
        # Glicko skips an "unknown" division before reaching the unplayed branch, and its
        # classifier maps catchweight to unknown — so any real division will do here.
        wc = glicko_service._classify_weight_class(p.weight_class)
        fight_map[p.id] = {
            "id": p.id, "date": when, "red_id": p.red_fighter_id, "blue_id": p.blue_fighter_id,
            "winner_id": None, "method": "", "weight_class": wc if wc != "unknown" else "lightweight",
            "finish_round": None, "finish_time_seconds": 0, "round_minutes": [5, 5, 5],
            "max_rounds": 3, "is_title": False, "is_5rd": False,
        }
    snaps = glicko_service.run_glicko_inmemory_cached(
        fight_map, rounds_by_fight, fighter_info, baselines,
        pre_ufc=glicko_service.sherdog_pre_ufc_records(fight_map))

    per_all = pd.concat([method_ratings.compute(df), _per_fighter(df, rd)], axis=1)
    per_all = per_all.loc[:, ~per_all.columns.duplicated()]
    keyed = pd.concat([df[["fight_id", "stats_fighter_id"]], per_all], axis=1)
    per = keyed[keyed["fight_id"] < 0].drop(columns="fight_id").set_index("stats_fighter_id")

    feat = build_features(df.copy(), rd, glicko_snapshots=snaps)
    cols = winner_feature_columns(feat)
    fight_cols = [c for c in cols if _is_fight_level(c)]
    fighter_cols = [c for c in cols if not _is_fight_level(c)]
    raw_cols = [c for c in WINNER_RAW_COLS if c in fighter_cols]
    raw_cols += [c for c in fighter_cols if c.startswith("glicko_")]

    probe_rows = feat[feat["fight_id"] < 0].set_index("stats_fighter_id")
    rows = probe_rows[fighter_cols + ["fighter_style"]].copy()
    rows["last_division"] = [
        _classify_weight_class(last_wc[f]) if f in last_wc else None for f in rows.index]
    rows["n_ufc_fights"] = [n_fights.get(f, 0) for f in rows.index]
    rows["div_run"] = [float(div_run.get(f, 0)) for f in rows.index]
    rows.index = rows.index.map(int)
    per.index = per.index.map(int)

    n_styles = int(feat["fighter_style"].max()) + 1
    state = State(
        built_at=datetime.now(), as_of=when, rows=rows, per=per,
        fighter_cols=fighter_cols, raw_cols=raw_cols, fight_cols=fight_cols,
        style_adv=_style_table(feat, n_styles),
        meta={"seconds": round(time.time() - t0, 1), "fighters": len(rows)},
    )
    log.info(f"  hypothetical state: {len(rows)} fighters in {state.meta['seconds']}s")
    return state


# In-memory state, a single-flight rebuild, and the on-disk copy so a restart is warm.
_state: State | None = None
_lock = threading.Lock()
_building = False
_last_error: str | None = None


def _path():
    from app.services.ufc.model import MODEL_DIR
    return MODEL_DIR / STATE_FILE


def save_state(state: State) -> None:
    tmp = _path().with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    tmp.replace(_path())


def _load_disk() -> State | None:
    try:
        with open(_path(), "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, AttributeError):
        return None


def refresh(blocking: bool = False) -> None:
    """Rebuild the state (in a background thread unless `blocking`). One build at a time."""
    global _building

    def run():
        global _state, _building, _last_error
        try:
            st = build_state()
            save_state(st)
            _state, _last_error = st, None
        except Exception as e:  # keep serving the previous state
            log.exception("hypothetical state build failed")
            _last_error = f"{type(e).__name__}: {e}"
        finally:
            _building = False

    with _lock:
        if _building:
            return
        _building = True
    if blocking:
        run()
        if _last_error:  # the scheduled job records failures; the API keeps serving
            raise RuntimeError(_last_error)
    else:
        threading.Thread(target=run, name="hypothetical-state", daemon=True).start()


def get_state() -> State:
    """The current state. Starts a rebuild when it is missing or a day old; raises
    NotReady only when there is nothing at all to serve meanwhile."""
    global _state
    if _state is None:
        _state = _load_disk()
    if _state is None or datetime.now() - _state.built_at > MAX_AGE:
        refresh()
    if _state is None:
        raise NotReady(_last_error or "building fighter features")
    return _state


def status() -> dict:
    st = _state or _load_disk()
    return {"ready": st is not None, "building": _building, "error": _last_error,
            "built_at": st.built_at.isoformat(timespec="seconds") if st else None,
            "as_of": str(st.as_of) if st else None}


# ---------------------------------------------------------------------------
# Scoring a pair
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _models():
    from app.services.ufc import ensemble, method_v2, rounds_v1
    ens = ensemble.load()
    if ens is None:
        raise RuntimeError("no served winner ensemble (models/ufc/h2h/ensemble_v1.pkl)")
    return ens, method_v2.load(), rounds_v1.load()


def _fighter(st: State, fid: int) -> pd.Series:
    if fid not in st.rows.index:
        raise UnknownFighter(str(fid))
    return st.rows.loc[fid]


def _corner_values(st: State, me: pd.Series, opp: pd.Series, division: str) -> dict:
    """This fighter's per-fighter columns, with the opponent- and division-dependent ones
    recomputed for this bout. Mirrors build_features."""
    from app.services.ufc.model import DIVISION_ORDER

    v = {c: me.get(c, np.nan) for c in st.fighter_cols}
    v = {c: (float(x) if x is not None and x == x else np.nan) for c, x in v.items()}

    # Opponent
    if "elo_expected" in v:
        v["elo_expected"] = 1 / (1 + 10 ** ((float(opp["elo"]) - float(me["elo"])) / 400))
    if "style_matchup_adv" in v:
        v["style_matchup_adv"] = float(st.style_adv[int(me["fighter_style"]), int(opp["fighter_style"])])
    for c in v:
        if c.startswith("xs_"):
            v[c] = np.nan
    if "bio_is_home" in v:
        v["bio_is_home"] = np.nan

    # Division: relative to the fighter's last bout, as build_features does.
    prev = me.get("last_division")
    prev = prev if isinstance(prev, str) else None
    rank = DIVISION_ORDER.get(division, np.nan)
    prev_rank = DIVISION_ORDER.get(prev, np.nan) if prev else np.nan
    delta = rank - prev_rank
    div = {
        "division_rank": rank,
        "division_rank_delta": 0.0 if delta != delta else delta,
        "moved_up": float(delta == delta and delta > 0),
        "moved_down": float(delta == delta and delta < 0),
        "changed_division": float(prev is not None and division != prev),
        # The run of bouts at their last division carries on only if this bout is there too.
        "fights_in_current_division": float(me.get("div_run", 0.0)) if division == prev else 0.0,
        "is_catchweight": float(division == "catchweight"),
    }
    v.update({k: x for k, x in div.items() if k in v})
    return v


def _fight_level(st: State, division: str, title: bool, rounds: int) -> dict:
    from app.services.ufc.model import _scheduled_minutes
    tf = "5-5-5-5-5" if rounds == 5 else "5-5-5"
    out = {}
    for c in st.fight_cols:
        if c.startswith("div_"):
            out[c] = float(c[len("div_"):] == division)
        elif c == "is_title_fight":
            out[c] = float(title)
        elif c == "is_five_round":
            out[c] = float(rounds == 5)
        elif c == "scheduled_rounds":
            out[c] = float(rounds)
        elif c == "scheduled_minutes":
            out[c] = _scheduled_minutes(tf)
        else:  # xs_fight_* and the matchmaking flags: unknown for a bout nobody booked
            out[c] = np.nan
    return out


def _frame(st: State, red: pd.Series, blue: pd.Series, division: str, title: bool,
           rounds: int) -> pd.DataFrame:
    """One serving-matchup row (the columns build_serving_matchup emits) plus the
    red_/blue_ per-fighter columns method_v2 and rounds_v1 read."""
    r = _corner_values(st, red, blue, division)
    b = _corner_values(st, blue, red, division)
    row = {"fight_id": "hypothetical"}
    row.update({f"fight_{c}": x for c, x in _fight_level(st, division, title, rounds).items()})
    for c in st.fighter_cols:
        row[f"diff_{c}"] = r[c] - b[c]
    for c in st.raw_cols:
        row[f"red_{c}"], row[f"blue_{c}"] = r[c], b[c]
    for corner, fid in (("red", int(red.name)), ("blue", int(blue.name))):
        p = st.per.loc[fid] if fid in st.per.index else pd.Series(dtype=float)
        for c in st.per.columns:
            row[f"{corner}_{c}"] = float(p.get(c, np.nan))
    for c in ("odds_red_prob", "odds_blue_prob", "odds_diff", "odds_fav_is_red", "elo_vs_odds"):
        row[c] = np.nan
    row["fight_altitude_m"], row["fight_high_altitude"] = np.nan, np.nan
    return pd.DataFrame([row])


def _shap(ens, frame: pd.DataFrame, top: int = 20) -> list[dict]:
    """Top SHAP contributions from the CatBoost member, as the nightly job stores them."""
    from catboost import Pool
    member = next((m for m in ens.members if m.name == "catboost"), None)
    if member is None:
        return []
    X = ens._impute(frame)[member.features].to_numpy(float)
    sv = member.model.model.get_feature_importance(Pool(X), type="ShapValues")[0, :-1]
    order = np.argsort(-np.abs(sv))[:top]
    return [{"feature_name": member.features[i], "shap_value": round(float(sv[i]), 6),
             "abs_value": round(float(abs(sv[i])), 6), "feature_value": round(float(X[0, i]), 4)}
            for i in order]


def _method(model, frame: pd.DataFrame, frame_sw: pd.DataFrame, p_red: float) -> dict | None:
    """Six-way grid from method_v2, averaged over both corner orientations."""
    if model is None:
        return None
    from app.services.ufc.method_v2 import CLASS_NAMES, predict_frame
    a = predict_frame(model, frame, np.array([p_red])).iloc[0]
    s = predict_frame(model, frame_sw, np.array([1 - p_red])).iloc[0]
    cells = {}
    for side, other in (("red", "blue"), ("blue", "red")):
        for k in ("ko", "sub", "dec"):
            cells[f"{side}_{k}_prob"] = (a[f"{side}_{k}_prob"] + s[f"{other}_{k}_prob"]) / 2
    ko = cells["red_ko_prob"] + cells["blue_ko_prob"]
    sub = cells["red_sub_prob"] + cells["blue_sub_prob"]
    dec = cells["red_dec_prob"] + cells["blue_dec_prob"]
    k = int(np.argmax((ko, sub, dec)))
    r4 = lambda x: round(float(x), 4)
    return {"predicted_method": CLASS_NAMES[k], "confidence": r4((ko, sub, dec)[k]),
            "ko_prob": r4(ko), "sub_prob": r4(sub), "dec_prob": r4(dec),
            **{c: r4(x) for c, x in cells.items()}, "distance_prob": r4(dec)}


def _rounds(model, frame: pd.DataFrame, p_red: float, method: dict | None) -> dict | None:
    """Fight-length curve, anchored to the method grid — rounds_v1.generate_predictions
    for a single row."""
    if model is None or method is None:
        return None
    from app.services.ufc.method_v2 import orient
    from app.services.ufc.rounds_v1 import _scheduled, add_win_prob, anchor_cells, curves4, summarize
    sched = _scheduled(frame)
    views = [orient(frame, np.ones(1, bool)), orient(frame, np.zeros(1, bool))]
    add_win_prob(views, np.array([p_red]))
    views = [v.reindex(columns=model.base_features) for v in views]
    cells = np.array([[method[f"{s}_{k}_prob"] for s in ("red", "blue") for k in ("ko", "sub", "dec")]])
    sm = summarize(anchor_cells(curves4(model, views, sched), cells), sched)[0]
    r4 = lambda x: None if x is None else round(float(x), 4)
    return {"curve": sm["curve"], "p_end": [r4(sm[f"p_end_r{k}"]) for k in range(1, 6)],
            "p_decision": r4(sm["p_decision"]),
            "over": {"1.5": r4(sm["over_1_5"]), "2.5": r4(sm["over_2_5"]),
                     "3.5": r4(sm["over_3_5"]), "4.5": r4(sm["over_4_5"])},
            "expected_minutes": r4(sm["expected_minutes"]),
            "median_finish_minute": r4(sm["median_finish_minute"]),
            "peak_bin_start": r4(sm["peak_bin_start"])}


def predict(red_id: int, blue_id: int, division: str, title: bool = False,
            rounds: int | None = None) -> dict:
    """Winner, method and fight-length predictions for a bout that is not booked."""
    if division not in DIVISIONS:
        raise ValueError(f"unknown division {division!r}")
    if red_id == blue_id:
        raise ValueError("a fighter cannot fight themselves")
    rounds = rounds or (5 if title else 3)
    if rounds not in (3, 5):
        raise ValueError("rounds must be 3 or 5")

    st = get_state()
    red, blue = _fighter(st, red_id), _fighter(st, blue_id)
    ens, method_model, rounds_model = _models()

    # Both orientations in one batch, with every model column present up front: on a
    # one-row input the cost is per call and per column added, not per row. One thread,
    # because HistGradientBoosting otherwise starts its OpenMP pool for each of ~280 trees
    # (~1s a call) to score two rows.
    from threadpoolctl import threadpool_limits
    both = pd.concat([_frame(st, red, blue, division, title, rounds),
                      _frame(st, blue, red, division, title, rounds)], ignore_index=True)
    both = both.reindex(columns=list(dict.fromkeys([*both.columns, *ens.features])))
    with threadpool_limits(1):
        pr = ens.model_prob(both)
        p_red = float((pr[0] + 1 - pr[1]) / 2)
        frame, frame_sw = both.iloc[[0]].reset_index(drop=True), both.iloc[[1]].reset_index(drop=True)
        method = _method(method_model, frame, frame_sw, p_red)
        rounds_pred = _rounds(rounds_model, frame, p_red, method)
        shap = _shap(ens, frame)

    return {
        "weight_class": bout_name(division, title),
        "division": division,
        "scheduled_rounds": rounds,
        "as_of": str(st.as_of),
        "prediction": {
            "predicted_winner": "red" if p_red >= 0.5 else "blue",
            "confidence": round(abs(p_red - 0.5), 4),
            # No market for a bout that will not happen: the served number IS the model's.
            "red_prob": round(p_red, 4),
            "model_prob": round(p_red, 4),
            "va_prob_low": None, "va_prob_high": None,
        },
        "method_prediction": method,
        "round_prediction": rounds_pred,
        "shap_values": shap,
        "notes": _notes(red, blue, division),
        "weight_lbs": {"red": _num(red.get("weight_lbs")), "blue": _num(blue.get("weight_lbs"))},
    }


def _num(x):
    return None if x is None or x != x else float(x)


#: Listed-weight gap past which a bout is outside what the model has seen. UFC bouts are
#: one division at most (catchweights, moves up); the trees flatten past that range, so a
#: 50 lb gap scores about the same as a 15 lb one.
SIZE_GAP_LBS = 20.0


def _notes(red: pd.Series, blue: pd.Series, division: str) -> list[str]:
    """What the reader should know about how far this bout is from anything the model
    was trained on."""
    out = []
    w = [float(f.get("weight_lbs", np.nan)) for f in (red, blue)]
    if all(x == x for x in w) and abs(w[0] - w[1]) >= SIZE_GAP_LBS:
        out.append("size_gap")
    for f in (red, blue):
        prev = f.get("last_division")
        if not isinstance(prev, str):
            out.append("debut")
        elif prev != division:
            out.append("division_change")
        if float(f.get("days_since_last", 0) or 0) > 3 * 365:
            out.append("long_layoff")
    return sorted(set(out))


# ---------------------------------------------------------------------------
# The fighter index for the picker
# ---------------------------------------------------------------------------

def fighter_index(db) -> list[dict]:
    """Every fighter with at least one completed UFC bout: identity, record, current
    division (their latest bout's), divisional rank and whether they are active."""
    from sqlalchemy import func, or_

    from app.models.ufc import UFCFight, UFCFighter, UFCFighterRanking
    from app.services.ufc.model import _classify_weight_class

    bouts = (
        db.query(UFCFight.red_fighter_id, UFCFight.blue_fighter_id, UFCFight.weight_class,
                 UFCFight.date)
        .filter(or_(UFCFight.winner_id.isnot(None), func.coalesce(UFCFight.method, "") != ""))
        .order_by(UFCFight.date, UFCFight.id)
        .all()
    )
    last: dict[int, tuple] = {}
    real_div: dict[int, str] = {}
    n: dict[int, int] = {}
    female: set[int] = set()
    opponents: dict[int, set] = {}
    for r, b, wc, d in bouts:
        div = _classify_weight_class(wc)
        for fid, opp in ((r, b), (b, r)):
            last[fid] = (div, d)
            n[fid] = n.get(fid, 0) + 1
            opponents.setdefault(fid, set()).add(opp)
            if div not in ("catchweight", "unknown"):
                real_div[fid] = div   # the last REAL division, so a catchweight bout
                                      # does not file a fighter under no division at all
            if is_womens(div):
                female.add(fid)
    # A woman whose only bouts were catchweights never fought in a women's division;
    # her opponents did. Bouts are never mixed, so one female opponent settles it.
    for fid, opps in opponents.items():
        if fid not in female and opps & female:
            female.add(fid)
    ranks = {
        int(fid): (wc, rank) for fid, wc, rank in
        db.query(UFCFighterRanking.fighter_id, UFCFighterRanking.weight_class, UFCFighterRanking.rank)
        .filter(UFCFighterRanking.rank > 0)
    }
    today = date.today()
    out = []
    # Named columns, not the entity: a whole UFCFighter row carries its cached portrait
    # bytes, and this reads thousands of fighters.
    F = UFCFighter
    cols = (F.id, F.first_name, F.last_name, F.nickname, F.country_code, F.image_url,
            F.wins, F.losses, F.draws, F.height, F.reach, F.stance, F.dob, F.status,
            F.image_data.isnot(None).label("cached"))
    for f in db.query(*cols).filter(F.id.in_(list(last))):
        fid = int(f.id)
        div = real_div.get(fid) or last[fid][0]
        rk = ranks.get(fid)
        age = None
        if isinstance(f.dob, date):
            age = today.year - f.dob.year - ((today.month, today.day) < (f.dob.month, f.dob.day))
        out.append({
            "id": str(fid), "first_name": f.first_name, "last_name": f.last_name,
            "nickname": f.nickname, "country_code": f.country_code,
            "image_url": f.image_url, "has_image": bool(f.cached or f.image_url),
            "wins": f.wins or 0, "losses": f.losses or 0, "draws": f.draws or 0,
            "height": f.height, "reach": f.reach, "stance": f.stance, "age": age,
            "status": f.status, "division": div,
            # Men's and women's divisions are separate ladders; the picker and the
            # predict route never match across them.
            "sex": "F" if fid in female else "M",
            "last_fight": str(last[fid][1]) if last[fid][1] else None,
            "ufc_fights": n.get(fid, 0),
            "rank": rk[1] if rk and rk[0] == div else None,
        })
    out.sort(key=lambda x: (x["rank"] is None, x["rank"] or 0, x["last_name"] or ""))
    return out


def fighter_index_cached(db) -> list[dict]:
    """fighter_index through the API's response cache, shared with /hypothetical/fighters."""
    from app.routers.ufc import RANKINGS_TTL
    from app.services import response_cache
    from app.services.response_cache import with_session
    return response_cache.cached("hypothetical:fighters", with_session(fighter_index), RANKINGS_TTL)


def default_division(red_div: str | None, blue_div: str | None) -> str:
    """The heavier fighter's division: the lighter one moving up is the usual superfight."""
    from app.services.ufc.model import DIVISION_ORDER
    divs = [d for d in (red_div, blue_div) if d in DIVISION_ORDER]
    if not divs:
        return "catchweight"
    return max(divs, key=lambda d: DIVISION_ORDER[d])


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="rebuild and save the feature state")
    a = ap.parse_args()
    if a.build:
        # Through the package path so the pickled State is importable by the app.
        from app.services.ufc import hypothetical as _h
        _h.save_state(_h.build_state())
