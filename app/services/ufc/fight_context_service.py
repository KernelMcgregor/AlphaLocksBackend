"""Matchup context for a single fight.

Serves the pre-fight data that has no route of its own: the 15-dimension Glicko
snapshot for each corner, the ranked skill differentials between them, finish rates,
layoff, division moves, and shared opponents.

Deliberately narrow. Career stats, fight logs and round-by-round stats already have
endpoints (`/fighters/{id}/career-stats`, `/fighters/{id}/fights`, `/fighters/{id}/stats`)
which the frontend caches and shares with the fighter profile page, so duplicating
them here would just be a second copy to keep in sync.

This module owns the per-fighter DB reads that `preview_service` used to define and
now imports. The dependency runs this way round on purpose: assembling matchup data
is a plain database concern, and a read-only endpoint should not have to import the
OpenAI client to serve it.
"""

from datetime import date

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.ufc import (
    UFCFight, UFCFighter, UFCFighterRanking, UFCGlickoSnapshot,
)


# ---------------------------------------------------------------------------
# Per-fighter reads shared with preview_service
# ---------------------------------------------------------------------------
GLICKO_DIMS = ["pts", "ko", "kod", "sub", "subd", "td", "tdd", "ctrl",
                "str_vol", "str_acc", "str_def", "dist", "clinch", "gnd", "durability"]


GLICKO_LABELS = {
    "pts": "Round Winning", "ko": "KO Power", "kod": "KO Defense",
    "sub": "Submission Offense", "subd": "Submission Defense",
    "td": "Takedown Offense", "tdd": "Takedown Defense",
    "ctrl": "Control Time", "str_vol": "Strike Volume",
    "str_acc": "Strike Accuracy", "str_def": "Strike Defense",
    "dist": "Distance Striking", "clinch": "Clinch Striking",
    "gnd": "Ground Striking", "durability": "Durability",
}


def _finish_rates(db: Session, fighter_id: int) -> dict:
    """Calculate KO and submission finish rates from all wins."""
    wins = (
        db.query(UFCFight)
        .filter(UFCFight.winner_id == fighter_id)
        .all()
    )
    total = len(wins)
    if total == 0:
        return {"ko_rate": 0, "sub_rate": 0, "total_wins": 0}
    ko_count = sum(1 for w in wins if w.method and "KO" in w.method.upper())
    sub_count = sum(1 for w in wins if w.method and "SUB" in w.method.upper())
    return {
        "ko_rate": round(ko_count / total, 2),
        "sub_rate": round(sub_count / total, 2),
        "total_wins": total,
    }


def _days_since_last_fight(db: Session, fighter_id: int) -> int | None:
    """Days between today and the fighter's most recent completed fight."""
    from datetime import date
    last = (
        db.query(UFCFight.date)
        .filter(
            or_(UFCFight.red_fighter_id == fighter_id, UFCFight.blue_fighter_id == fighter_id),
            UFCFight.winner_id.isnot(None),
        )
        .order_by(UFCFight.date.desc())
        .first()
    )
    if not last or not last[0]:
        return None
    fight_date = last[0] if isinstance(last[0], date) else date.fromisoformat(str(last[0]))
    return (date.today() - fight_date).days


def _division_change(db: Session, fighter_id: int, current_weight_class: str | None) -> dict:
    """Check if this is a UFC debut or division change."""
    past_fights = (
        db.query(UFCFight.weight_class)
        .filter(
            or_(UFCFight.red_fighter_id == fighter_id, UFCFight.blue_fighter_id == fighter_id),
            UFCFight.winner_id.isnot(None),
        )
        .order_by(UFCFight.date.desc())
        .limit(5)
        .all()
    )
    if not past_fights:
        return {"ufc_debut": True, "division_change": False, "previous_division": None}
    prev_class = past_fights[0][0]
    changed = (
        current_weight_class is not None
        and prev_class is not None
        and current_weight_class.strip().lower() != prev_class.strip().lower()
    )
    return {
        "ufc_debut": False,
        "division_change": changed,
        "previous_division": prev_class if changed else None,
    }


def _scheduled_rounds(fight) -> int | None:
    """Number of rounds the bout is scheduled for, from `time_format`.

    The stored format is a dash-joined list of round lengths in minutes: '5-5-5' is
    a three-rounder, '5-5-5-5-5' a championship five. Untimed bouts are stored as
    'No Time Limit' and have no round count.

    This used to parse a leading round count ('3 Rnd (5-5-5)'), a shape that appears
    nowhere in the table -- so it returned None for every fight, including in the
    preview prompt. The leading-count branch is kept in case the scraper ever emits
    it, but the dash form is what the data actually holds.
    """
    fmt = (fight.time_format or "").strip()
    if not fmt:
        return None

    head = fmt.split()
    if head and head[0].isdigit() and len(head) > 1:
        return int(head[0])

    segments = fmt.split("-")
    if all(s.strip().isdigit() for s in segments):
        # Guards against a malformed row claiming an implausible round count.
        return len(segments) if 1 <= len(segments) <= 5 else None
    return None


def _percentile_tier(pct: float) -> str:
    if pct >= 90:
        return "Elite"
    if pct >= 70:
        return "Strong"
    if pct >= 40:
        return "Average"
    if pct >= 20:
        return "Below Avg"
    return "Weak"


def _get_glicko_data(db: Session, fight_id: int, fighter_id: int, weight_class: str | None) -> dict | None:
    """Get Glicko snapshot and ranking data for a fighter."""
    snapshot = (
        db.query(UFCGlickoSnapshot)
        .filter(UFCGlickoSnapshot.fight_id == fight_id, UFCGlickoSnapshot.fighter_id == fighter_id)
        .first()
    )
    # An upcoming bout has no snapshot of its own — glicko_service writes one row per
    # fighter per *completed* fight (the pre-fight state going into it), so a fight
    # that has not happened is not in that table and every skill box on the fight page
    # came back empty. The state a fighter carries into an announced bout is the
    # snapshot from their most recent completed fight, which is what this falls back
    # to. `carried_in` says so, rather than passing it off as this-bout data.
    carried_in = False
    if not snapshot:
        snapshot = (
            db.query(UFCGlickoSnapshot)
            .join(UFCFight, UFCFight.id == UFCGlickoSnapshot.fight_id)
            .filter(
                UFCGlickoSnapshot.fighter_id == fighter_id,
                UFCFight.date.isnot(None),
                UFCFight.winner_id.isnot(None),
            )
            .order_by(UFCFight.date.desc())
            .first()
        )
        carried_in = snapshot is not None

    # Filter on the division too. Fighters ranked in a p4p table have a second row, and
    # .first() on fighter_id alone returned whichever the DB happened to yield — so a
    # top-25 p4p rank could be shown as the divisional one. Also exclude rank=0
    # placeholder rows, same reason as get_rankings().
    from app.services.ufc.points_ranking_service import _classify_weight_class

    wc_key = _classify_weight_class(weight_class) if weight_class else None
    ranking = (
        db.query(UFCFighterRanking)
        .filter(
            UFCFighterRanking.fighter_id == fighter_id,
            UFCFighterRanking.weight_class == wc_key,
            UFCFighterRanking.rank > 0,
        )
        .first()
    ) if wc_key and wc_key != "unknown" else None

    # feature_profile stores bare floats already normalised to 0-99 within the division.
    # This previously expected {dim: {"percentile": n}}, which no writer has ever
    # produced, so `percentiles` was always empty and every percentile/tier below was
    # None — the whole block rendered blank.
    import json
    percentiles = {}
    if ranking and ranking.feature_profile:
        try:
            profile = json.loads(ranking.feature_profile)
            percentiles = {
                k: float(v) for k, v in profile.items()
                if k in GLICKO_DIMS and isinstance(v, (int, float))
            }
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            pass

    # Percentiles come from the *current* ranking profile and ratings from the
    # snapshot, so a fighter with a ranking but no snapshot at all (a debutant, or
    # before the first replay) still gets a plottable percentile profile.
    if snapshot is None and not percentiles:
        return None

    ratings = {}
    for dim in GLICKO_DIMS:
        val = getattr(snapshot, dim, None) if snapshot is not None else None
        pct = percentiles.get(dim)
        ratings[dim] = {
            "label": GLICKO_LABELS[dim],
            "rating": round(val, 1) if val is not None else None,
            "percentile": round(pct, 1) if pct is not None else None,
            "tier": _percentile_tier(pct) if pct is not None else None,
        }

    result = {"dimensions": ratings, "ratings_carried_in": carried_in}
    if ranking:
        result["division_rank"] = ranking.rank
        # `score` is a 0-1000 division-normalised points score, NOT a win rate. The old
        # `score * 100` label read as a percentage and rendered values up to 100000%.
        result["division_score"] = round(ranking.score, 1)

    return result


def glicko_matchup_edges(red_glicko: dict | None, blue_glicko: dict | None) -> list[dict]:
    """Rank the Glicko dimensions by how far apart the two corners sit.

    Returns dicts rather than a formatted string so the fight-context endpoint can
    serve the same ranking the preview prompt is built from. `_glicko_edge_block`
    below is the prompt's view of this list.
    """
    if not red_glicko or not blue_glicko:
        return []

    edges = []
    for dim in GLICKO_DIMS:
        rd = red_glicko["dimensions"].get(dim, {})
        bd = blue_glicko["dimensions"].get(dim, {})
        if rd.get("rating") is not None and bd.get("rating") is not None:
            edges.append({
                "dim": dim,
                "label": GLICKO_LABELS[dim],
                "red_rating": rd["rating"],
                "blue_rating": bd["rating"],
                "diff": round(rd["rating"] - bd["rating"], 1),
                "red_percentile": rd.get("percentile"),
                "blue_percentile": bd.get("percentile"),
            })

    edges.sort(key=lambda e: abs(e["diff"]), reverse=True)
    return edges


# ---------------------------------------------------------------------------
# The matchup payload
# ---------------------------------------------------------------------------
def _age(fighter: UFCFighter) -> int | None:
    if not fighter.dob:
        return None
    try:
        today = date.today()
        dob = fighter.dob if isinstance(fighter.dob, date) else date.fromisoformat(str(fighter.dob))
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except (ValueError, TypeError):
        return None


def _completed_bouts(db: Session, fighter_id: int) -> list[UFCFight]:
    return (
        db.query(UFCFight)
        .filter(
            or_(UFCFight.red_fighter_id == fighter_id, UFCFight.blue_fighter_id == fighter_id),
            UFCFight.winner_id.isnot(None),
        )
        .order_by(UFCFight.date.desc())
        .all()
    )


def _bout_summary(fight: UFCFight, fighter_id: int) -> dict:
    return {
        "fight_id": fight.id,
        "date": str(fight.date) if fight.date else None,
        "won": fight.winner_id == fighter_id,
        "method": (fight.method or "").strip() or None,
        "round": fight.finish_round,
    }


def _common_opponents(db: Session, red_id: int, blue_id: int) -> list[dict]:
    """Opponents both fighters have already faced, most recent meeting first.

    Only completed bouts count, and the fighters themselves are excluded -- a
    previous meeting between these two is a rematch, not a common opponent.
    """
    def by_opponent(fighter_id):
        out = {}
        for f in _completed_bouts(db, fighter_id):
            opp_id = f.blue_fighter_id if f.red_fighter_id == fighter_id else f.red_fighter_id
            # _completed_bouts is already newest-first, so the first row wins and
            # the most recent meeting is the one reported.
            out.setdefault(opp_id, f)
        return out

    red_bouts = by_opponent(red_id)
    blue_bouts = by_opponent(blue_id)
    shared = (set(red_bouts) & set(blue_bouts)) - {red_id, blue_id}
    if not shared:
        return []

    names = {
        f.id: f"{f.first_name} {f.last_name}"
        for f in db.query(UFCFighter).filter(UFCFighter.id.in_(shared)).all()
    }

    rows = [
        {
            "opponent_id": opp_id,
            "opponent_name": names.get(opp_id, "Unknown"),
            "red": _bout_summary(red_bouts[opp_id], red_id),
            "blue": _bout_summary(blue_bouts[opp_id], blue_id),
        }
        for opp_id in shared
    ]
    # Sort by the more recent of the two meetings, newest first. A missing date
    # sorts last rather than raising on the None comparison.
    rows.sort(key=lambda r: max(r["red"]["date"] or "", r["blue"]["date"] or ""), reverse=True)
    return rows


def _corner(db: Session, fight: UFCFight, fighter: UFCFighter) -> dict:
    return {
        "fighter_id": fighter.id,
        "age": _age(fighter),
        # Pre-fight ratings; the percentiles come from the fighter's CURRENT
        # divisional profile, so for a past fight they read "as of today".
        "glicko": _get_glicko_data(db, fight.id, fighter.id, fight.weight_class),
        "finish_rates": _finish_rates(db, fighter.id),
        "days_since_last_fight": _days_since_last_fight(db, fighter.id),
        "division_info": _division_change(db, fighter.id, fight.weight_class),
    }


def gather_matchup_context(fight_id: int, db: Session) -> dict | None:
    """Assemble the matchup payload. None when the fight or either corner is missing."""
    fight = db.query(UFCFight).filter(UFCFight.id == fight_id).first()
    if not fight:
        return None

    red = db.query(UFCFighter).filter(UFCFighter.id == fight.red_fighter_id).first()
    blue = db.query(UFCFighter).filter(UFCFighter.id == fight.blue_fighter_id).first()
    if not red or not blue:
        return None

    red_ctx = _corner(db, fight, red)
    blue_ctx = _corner(db, fight, blue)

    return {
        "fight_id": fight.id,
        "weight_class": fight.weight_class,
        "scheduled_rounds": _scheduled_rounds(fight),
        "red": red_ctx,
        "blue": blue_ctx,
        "edges": glicko_matchup_edges(red_ctx["glicko"], blue_ctx["glicko"]),
        "common_opponents": _common_opponents(db, red.id, blue.id),
    }
