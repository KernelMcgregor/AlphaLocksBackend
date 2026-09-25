import json

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import or_
from sqlalchemy.orm import Session, aliased, joinedload

from app.database import get_db
from app.services import response_cache
from app.services.response_cache import with_session
from app.models.ufc import (
    UFCEvent, UFCFight, UFCFighter, UFCFighterCareerStats, UFCFighterSimilarity,
    UFCFightOdds, UFCMethodOdds,
    UFCFightPrediction, UFCFightPreview, UFCMethodPrediction, UFCFightShapValue, UFCFightStats,
    UFCRankingHistory,
)
from app.schemas.ufc import (
    UFCEventDetailResponse,
    UFCRankHistoryPoint,
    UFCEventResponse,
    UFCFighterCareerStatsResponse,
    UFCFightDetailResponse,
    UFCFightResponse,
    UFCFighterResponse,
    UFCFightStatsResponse,
    UFCSimilarFighterResponse,
)

router = APIRouter(prefix="/ufc", tags=["ufc"])

# Server-side cache lifetimes (seconds) for the views that are slow to build — see
# app/services/response_cache.py. Past the TTL the old answer is still served while a
# background refresh runs, so these bound staleness, not latency.
UPCOMING_TTL = 5 * 60
RANKINGS_TTL = 10 * 60
PREVIEWS_TTL = 10 * 60
FIGHT_TTL = 5 * 60


# --- Fighters ---

@router.get("/fighters", response_model=list[UFCFighterResponse])
def list_fighters(
    weight_class: str | None = None,
    search: str | None = None,
    limit: int = Query(default=50, le=500),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(UFCFighter)
    if search:
        query = query.filter(
            (UFCFighter.first_name.ilike(f"%{search}%"))
            | (UFCFighter.last_name.ilike(f"%{search}%"))
            | (UFCFighter.nickname.ilike(f"%{search}%"))
        )
    return query.order_by(UFCFighter.last_name).offset(offset).limit(limit).all()


@router.get("/fighters/{fighter_id}", response_model=UFCFighterResponse)
def get_fighter(fighter_id: int, db: Session = Depends(get_db)):
    fighter = db.get(UFCFighter, fighter_id)
    if not fighter:
        raise HTTPException(status_code=404, detail="Fighter not found")
    return fighter


@router.get("/fighters/{fighter_id}/image")
def get_fighter_image(fighter_id: int, db: Session = Depends(get_db)):
    """Serve a fighter's headshot.

    Cached bytes when scripts/cache_fighter_images.py has stored them, otherwise a
    redirect to the source URL. One URL works either way, so callers do not have to know
    whether the cache has been populated — and when a UFC.com `?itok=` signature
    eventually expires, the cached copy is what keeps the portrait alive.
    """
    row = (
        db.query(UFCFighter.image_data, UFCFighter.image_mime, UFCFighter.image_url)
        .filter(UFCFighter.id == fighter_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Fighter not found")

    image_data, image_mime, image_url = row
    if image_data:
        return Response(
            content=image_data,
            media_type=image_mime or "image/png",
            headers={
                # Immutable in practice: a new portrait arrives as a new URL, and the
                # cache is refreshed by an explicit script run.
                "Cache-Control": "public, max-age=604800",
                "ETag": f'W/"{fighter_id}-{len(image_data)}"',
            },
        )
    if image_url:
        return RedirectResponse(image_url, status_code=302)
    raise HTTPException(status_code=404, detail="No image for fighter")


@router.get("/fighters/{fighter_id}/fights", response_model=list[UFCFightResponse])
def get_fighter_fights(fighter_id: int, db: Session = Depends(get_db)):
    return (
        db.query(UFCFight)
        .filter((UFCFight.red_fighter_id == fighter_id) | (UFCFight.blue_fighter_id == fighter_id))
        .order_by(UFCFight.id.desc())
        .all()
    )


@router.get("/fighters/{fighter_id}/stats", response_model=list[UFCFightStatsResponse])
def get_fighter_stats(fighter_id: int, db: Session = Depends(get_db)):
    return db.query(UFCFightStats).filter(UFCFightStats.fighter_id == fighter_id).all()


@router.get("/fighters/{fighter_id}/rank-history", response_model=list[UFCRankHistoryPoint])
def get_fighter_rank_history(fighter_id: int, db: Session = Depends(get_db)):
    """Divisional rank after each of this fighter's bouts.

    Reads the precomputed `ufc_ranking_history` table — see
    `app.services.ufc.rank_history_backfill` for why this cannot be computed per
    request. Returns [] if the backfill has not been run.

    Each of the fighter's bout dates is matched to the ranking published on or
    immediately after it, which is the standing that bout produced.
    """
    history = (
        db.query(UFCRankingHistory)
        .filter(UFCRankingHistory.fighter_id == fighter_id)
        .order_by(UFCRankingHistory.as_of)
        .all()
    )
    if not history:
        return []

    fights = (
        db.query(UFCFight)
        .filter(
            (UFCFight.red_fighter_id == fighter_id) | (UFCFight.blue_fighter_id == fighter_id)
        )
        .filter(UFCFight.date.isnot(None))
        # Completed bouts only. A scheduled fight has no winner and no method, and the
        # backfill covers announced future event dates, so without this an upcoming
        # bout would plot a rank the fighter has not earned yet. Draws and no-contests
        # are kept — they happened, they just have no winner.
        .filter(UFCFight.date <= dt.date.today())
        .filter(UFCFight.winner_id.isnot(None) | UFCFight.method.isnot(None))
        .order_by(UFCFight.date)
        .all()
    )

    # A bout on date D produces the ranking stamped D (the publish for that event).
    # A fighter is ranked in their division AND in p4p on the same date; the
    # divisional row is the one meant by "their rank", so p4p only acts as a
    # fallback for someone who somehow has no divisional row.
    by_date: dict = {}
    for h in history:
        is_p4p = h.weight_class.startswith("p4p")
        existing = by_date.get(h.as_of)
        if existing is None or (existing.weight_class.startswith("p4p") and not is_p4p):
            by_date[h.as_of] = h
    out: list[UFCRankHistoryPoint] = []
    for f in fights:
        h = by_date.get(f.date)
        if h is None:
            continue
        opponent = f.blue_fighter_id if f.red_fighter_id == fighter_id else f.red_fighter_id
        out.append(UFCRankHistoryPoint(
            as_of=h.as_of,
            weight_class=h.weight_class,
            rank=h.rank,
            score=h.score,
            total_ranked=h.total_ranked,
            fight_id=str(f.id),
            opponent_id=str(opponent) if opponent else None,
            won=(f.winner_id == fighter_id) if f.winner_id else None,
        ))
    return out


@router.get("/fighters/{fighter_id}/career-stats", response_model=UFCFighterCareerStatsResponse)
def get_fighter_career_stats(fighter_id: int, db: Session = Depends(get_db)):
    stats = db.query(UFCFighterCareerStats).filter(
        UFCFighterCareerStats.fighter_id == fighter_id
    ).first()
    if not stats:
        raise HTTPException(status_code=404, detail="Career stats not found for this fighter")
    return stats


@router.get("/fighters/{fighter_id}/similar", response_model=list[UFCSimilarFighterResponse])
def get_similar_fighters(
    fighter_id: int,
    limit: int = Query(default=10, le=20),
    same_division_only: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    """Stylistic comparables, nearest first.

    Cross-division by default: the features are within-division percentiles, so a
    flyweight and a heavyweight are directly comparable and the cross-division analogue
    is usually the more interesting answer. `same_division_only` narrows it without a
    second round trip because `same_division` is denormalised onto the row.

    Empty list rather than 404 when a fighter has no comparables — that is the normal
    state for anyone below the eligibility floor (3 decided fights / 10 rounds), not an
    error, and the panel just does not render.
    """
    rows = (
        db.query(UFCFighterSimilarity, UFCFighter)
        .join(UFCFighter, UFCFighter.id == UFCFighterSimilarity.similar_fighter_id)
        .filter(UFCFighterSimilarity.fighter_id == fighter_id)
    )
    if same_division_only:
        rows = rows.filter(UFCFighterSimilarity.same_division.is_(True))

    out = []
    for sim, fighter in rows.order_by(UFCFighterSimilarity.rank).limit(limit).all():
        try:
            drivers = json.loads(sim.top_drivers or "[]")
        except (TypeError, ValueError):
            drivers = []
        out.append(UFCSimilarFighterResponse(
            id=fighter.id,
            first_name=fighter.first_name,
            last_name=fighter.last_name,
            nickname=fighter.nickname,
            image_url=fighter.image_url,
            country_code=fighter.country_code,
            wins=fighter.wins,
            losses=fighter.losses,
            draws=fighter.draws,
            rank=sim.rank,
            similarity=sim.similarity,
            same_division=sim.same_division,
            top_drivers=drivers,
            previous_rank=sim.previous_rank,
        ))
    return out


@router.get("/career-stats", response_model=list[UFCFighterCareerStatsResponse])
def list_career_stats(
    limit: int = Query(default=500, le=2000),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    return (
        db.query(UFCFighterCareerStats)
        .order_by(UFCFighterCareerStats.fighter_id)
        .offset(offset)
        .limit(limit)
        .all()
    )


# --- Events ---

@router.get("/events", response_model=list[UFCEventResponse])
def list_events(
    limit: int = Query(default=50, le=500),
    offset: int = 0,
    ufc_only: bool = Query(default=True),
    db: Session = Depends(get_db),
):
    query = db.query(UFCEvent)
    if ufc_only:
        # Filter to events that have fights (excludes stub events from fighter profile scraping)
        from sqlalchemy import exists
        query = query.filter(
            exists().where(UFCFight.event_id == UFCEvent.id)
        )
    return query.order_by(UFCEvent.date.desc()).offset(offset).limit(limit).all()


@router.get("/events/{event_id}", response_model=UFCEventResponse)
def get_event(event_id: int, db: Session = Depends(get_db)):
    event = db.get(UFCEvent, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@router.get("/events/{event_id}/fights", response_model=list[UFCFightResponse])
def get_event_fights(event_id: int, db: Session = Depends(get_db)):
    return db.query(UFCFight).filter(UFCFight.event_id == event_id).all()


@router.get("/events/{event_id}/detail", response_model=UFCEventDetailResponse)
def get_event_detail(event_id: int, db: Session = Depends(get_db)):
    event = (
        db.query(UFCEvent)
        .filter(UFCEvent.id == event_id)
        .first()
    )
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    fights = (
        db.query(UFCFight)
        .options(
            joinedload(UFCFight.red_fighter),
            joinedload(UFCFight.blue_fighter),
            joinedload(UFCFight.winner),
            joinedload(UFCFight.stats),
        )
        .filter(UFCFight.event_id == event_id)
        .order_by(UFCFight.card_position.is_(None), UFCFight.card_position, UFCFight.id)
        .all()
    )
    # Attach consensus odds (first bookmaker found) to each fight
    fight_ids = [f.id for f in fights]
    odds_rows = db.query(UFCFightOdds).filter(UFCFightOdds.fight_id.in_(fight_ids)).all() if fight_ids else []
    odds_map = {}
    for o in odds_rows:
        if o.fight_id not in odds_map:  # keep first (consensus/primary)
            odds_map[o.fight_id] = o

    fight_dicts = []
    for f in fights:
        fd = {
            "id": f.id, "ufcstats_id": f.ufcstats_id, "date": f.date,
            "event_id": f.event_id, "red_fighter_id": f.red_fighter_id,
            "blue_fighter_id": f.blue_fighter_id, "winner_id": f.winner_id,
            "red_result": f.red_result, "blue_result": f.blue_result,
            "weight_class": f.weight_class, "method": f.method,
            "finish_round": f.finish_round, "finish_time": f.finish_time,
            "details": f.details, "referee": f.referee,
            "created_at": f.created_at, "updated_at": f.updated_at,
            "red_fighter": f.red_fighter, "blue_fighter": f.blue_fighter,
            "winner": f.winner, "stats": f.stats,
        }
        o = odds_map.get(f.id)
        if o:
            fd["red_odds"] = o.red_odds
            fd["blue_odds"] = o.blue_odds
        fight_dicts.append(fd)

    return {"id": event.id, "ufcstats_id": event.ufcstats_id, "name": event.name,
            "date": event.date, "location": event.location,
            "created_at": event.created_at, "fights": fight_dicts}


# --- Fights ---

@router.get("/fights", response_model=list[UFCFightResponse])
def list_fights(
    limit: int = Query(default=50, le=500),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    return db.query(UFCFight).order_by(UFCFight.id.desc()).offset(offset).limit(limit).all()


@router.get("/fights/{fight_id}")
def get_fight(fight_id: int):
    # ~1.8s cold (odds, SHAP, stats, markets — a dozen remote round trips), and every
    # fight page and preview page opens with it.
    return response_cache.cached(
        f"fight:{fight_id}", with_session(_build_fight, fight_id), FIGHT_TTL,
    )


def _build_fight(db: Session, fight_id: int):
    fight = (
        db.query(UFCFight)
        .options(
            joinedload(UFCFight.red_fighter),
            joinedload(UFCFight.blue_fighter),
            joinedload(UFCFight.winner),
            joinedload(UFCFight.stats),
        )
        .filter(UFCFight.id == fight_id)
        .first()
    )
    if not fight:
        raise HTTPException(status_code=404, detail="Fight not found")

    # Add prediction
    pred = db.query(UFCFightPrediction).filter(UFCFightPrediction.fight_id == fight_id).first()
    # Add method prediction
    method_pred = db.query(UFCMethodPrediction).filter(UFCMethodPrediction.fight_id == fight_id).first()
    # Add odds (all bookmakers)
    odds_rows = db.query(UFCFightOdds).filter(UFCFightOdds.fight_id == fight_id).all()
    # Add SHAP values
    shap_rows = db.query(UFCFightShapValue).filter(UFCFightShapValue.fight_id == fight_id).order_by(UFCFightShapValue.abs_value.desc()).all()

    # Add event info
    event = db.query(UFCEvent).filter(UFCEvent.id == fight.event_id).first()

    result = UFCFightDetailResponse.model_validate(fight).model_dump()
    result["event"] = {
        "name": event.name,
        "date": str(event.date),
        "location": event.location,
    } if event else None
    result["prediction"] = {
        "predicted_winner": pred.predicted_winner,
        "confidence": pred.confidence,
        "red_prob": pred.red_prob,
        # Venn-Abers calibration bounds. Stored since the model was calibrated but
        # never serialized, so the UI had no way to show how wide the interval is.
        "va_prob_low": pred.va_prob_low,
        "va_prob_high": pred.va_prob_high,
    } if pred else None
    result["method_prediction"] = {
        "predicted_method": method_pred.predicted_method,
        "confidence": method_pred.confidence,
        "ko_prob": method_pred.ko_prob,
        "sub_prob": method_pred.sub_prob,
        "dec_prob": method_pred.dec_prob,
    } if method_pred else None
    result["odds"] = [{
        "bookmaker": o.bookmaker,
        "red_odds": o.red_odds,
        "blue_odds": o.blue_odds,
        # Vig-inclusive implied probabilities, as scraped. The frontend used to
        # re-derive these from the American odds; they are stored, so serve them.
        "red_implied_prob": o.red_implied_prob,
        "blue_implied_prob": o.blue_implied_prob,
        "updated_at": o.updated_at.isoformat() if o.updated_at else None,
    } for o in odds_rows]
    result["shap_values"] = [{
        "feature_name": s.feature_name,
        "shap_value": s.shap_value,
        "abs_value": s.abs_value,
        "feature_value": s.feature_value,
    } for s in shap_rows]

    # Add method odds (Bovada)
    method_odds_row = db.query(UFCMethodOdds).filter(UFCMethodOdds.fight_id == fight_id).first()
    result["method_odds"] = {
        "bookmaker": method_odds_row.bookmaker,
        "ko_odds": method_odds_row.ko_odds,
        "sub_odds": method_odds_row.sub_odds,
        "dec_odds": method_odds_row.dec_odds,
        "ko_prob": method_odds_row.ko_prob,
        "sub_prob": method_odds_row.sub_prob,
        "dec_prob": method_odds_row.dec_prob,
        "red_ko_odds": method_odds_row.red_ko_odds,
        "red_sub_odds": method_odds_row.red_sub_odds,
        "red_dec_odds": method_odds_row.red_dec_odds,
        "blue_ko_odds": method_odds_row.blue_ko_odds,
        "blue_sub_odds": method_odds_row.blue_sub_odds,
        "blue_dec_odds": method_odds_row.blue_dec_odds,
    } if method_odds_row else None

    # Add preview
    preview = db.query(UFCFightPreview).filter(UFCFightPreview.fight_id == fight_id).first()
    result["preview"] = {
        "content": preview.content,
        "model_used": preview.model_used,
        "generated_at": preview.created_at.isoformat() if preview.created_at else None,
    } if preview else None

    # Prediction-market quotes (Kalshi, Polymarket). Served from the same call as the odds board
    # because the page already renders both side by side; a second request would only add a
    # render pass where the exchange rows pop in after the sportsbook ones.
    from app.services.ufc.prediction_markets.serving import consensus_prob, fight_payload
    result["prediction_markets"] = fight_payload(db, fight_id) or None
    result["market_consensus"] = consensus_prob(db, fight_id)

    return result


@router.get("/previews")
def list_previews(
    limit: int = Query(default=60, le=200),
    offset: int = 0,
):
    """Every written preview as an index row — the list behind "See All Articles".

    Deliberately not the full article: the index only needs a headline and an
    opening line, and the bodies run to a few thousand words each. The reader
    follows the row to `/ufc/fights/{id}/preview`, which serves the whole piece
    from the fight payload that page already fetches.
    """
    return response_cache.cached(
        f"previews:{limit}:{offset}", with_session(_build_previews, limit, offset), PREVIEWS_TTL,
    )


def _build_previews(db: Session, limit: int, offset: int):
    red_f = aliased(UFCFighter)
    blue_f = aliased(UFCFighter)
    rows = (
        db.query(UFCFightPreview, UFCFight, UFCEvent, red_f, blue_f)
        .join(UFCFight, UFCFight.id == UFCFightPreview.fight_id)
        .outerjoin(UFCEvent, UFCEvent.id == UFCFight.event_id)
        .outerjoin(red_f, red_f.id == UFCFight.red_fighter_id)
        .outerjoin(blue_f, blue_f.id == UFCFight.blue_fighter_id)
        # Newest card first, and within a card the most recently written piece.
        .order_by(UFCFight.date.desc(), UFCFightPreview.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    def _headline_and_lede(content: str) -> tuple[str | None, str | None]:
        """First markdown heading, and the first paragraph that follows it."""
        headline = None
        lede = None
        for line in (content or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if headline is None:
                    headline = stripped.lstrip("#").strip()
                continue
            if headline is not None or lede is None:
                # Skip table rows and list markers — they read as noise in a card.
                if stripped.startswith(("|", "-", "*", ">")):
                    continue
                lede = stripped
                break
        return headline, lede

    out = []
    for preview, fight, event, red, blue in rows:
        headline, lede = _headline_and_lede(preview.content)
        out.append({
            "fight_id": str(fight.id),
            "headline": headline,
            "lede": lede,
            "weight_class": fight.weight_class,
            "fight_date": fight.date.isoformat() if fight.date else None,
            "is_upcoming": fight.winner_id is None,
            "event_name": event.name if event else None,
            "event_date": event.date.isoformat() if event and event.date else None,
            "event_location": event.location if event else None,
            "red_name": f"{red.first_name} {red.last_name}".strip() if red else None,
            "blue_name": f"{blue.first_name} {blue.last_name}".strip() if blue else None,
            "model_used": preview.model_used,
            "generated_at": preview.created_at.isoformat() if preview.created_at else None,
        })
    return out


@router.get("/fights/{fight_id}/context")
def get_fight_context(fight_id: int):
    """Matchup context: per-corner Glicko skills, skill edges, and shared opponents.

    Complements `GET /fights/{id}` rather than replacing it — see
    `app/services/ufc/fight_context_service.py` for what is deliberately left out.
    """
    return response_cache.cached(
        f"fight-context:{fight_id}", with_session(_build_fight_context, fight_id), FIGHT_TTL,
    )


def _build_fight_context(db: Session, fight_id: int):
    from app.services.ufc.fight_context_service import gather_matchup_context

    context = gather_matchup_context(fight_id, db)
    if not context:
        raise HTTPException(status_code=404, detail="Fight not found")
    return context


@router.get("/fights/{fight_id}/market-history")
def get_fight_market_history(
    fight_id: int,
    market_type: str = Query(default="moneyline"),
    db: Session = Depends(get_db),
):
    """Prediction-market price curves for a fight, one series per venue.

    Kept off the main fight payload deliberately: a curve is hundreds of points per venue, the
    fight page is already a large single response, and only the movement chart needs it.

    Points are red-corner probabilities. Both venues publish their own price history, so this is
    real historical data rather than a record of when we happened to poll — which is why curves
    exist for fights that settled long before any of this was built.
    """
    from app.services.ufc.prediction_markets.serving import market_history

    fight = db.query(UFCFight.id).filter(UFCFight.id == fight_id).first()
    if not fight:
        raise HTTPException(status_code=404, detail="Fight not found")
    return {"fight_id": fight_id, "market_type": market_type,
            "series": market_history(db, fight_id, market_type)}


# --- Predictions ---

@router.get("/model/metrics")
def get_model_metrics(db: Session = Depends(get_db)):
    """Get model performance metrics with pick-based and edge-based P/L."""
    from collections import defaultdict
    from sqlalchemy import and_

    def _american_to_decimal(odds):
        if odds > 0:
            return 1 + odds / 100
        return 1 + 100 / abs(odds)

    def _implied_prob(american_odds):
        if american_odds > 0:
            return 100 / (american_odds + 100)
        return abs(american_odds) / (abs(american_odds) + 100)

    # Get all predictions for decided fights
    preds = (
        db.query(UFCFightPrediction, UFCFight.winner_id, UFCFight.red_fighter_id, UFCFight.id)
        .join(UFCFight, UFCFightPrediction.fight_id == UFCFight.id)
        .filter(and_(UFCFight.winner_id.isnot(None), UFCFight.date >= "2015-01-01"))
        .all()
    )
    if not preds:
        return {"total": 0, "correct": 0, "accuracy": 0, "confidence_splits": [], "edge_splits": [], "fights": []}

    # Get all odds grouped by fight_id
    all_fight_ids = [fight_id for _, _, _, fight_id in preds]
    all_odds = db.query(UFCFightOdds).filter(UFCFightOdds.fight_id.in_(all_fight_ids)).all()
    odds_by_fight = defaultdict(list)
    for o in all_odds:
        odds_by_fight[o.fight_id].append(o)

    # Process each fight
    fight_data = []
    for pred, winner_id, red_fighter_id, fight_id in preds:
        red_won = winner_id == red_fighter_id
        pick_won = (pred.predicted_winner == "red") == red_won
        odds_rows = odds_by_fight.get(fight_id, [])

        # Pick P/L (using first available bookmaker odds)
        pick_pl = None
        pick_odds = None
        if odds_rows:
            o = odds_rows[0]
            picked_red = pred.predicted_winner == "red"
            pick_odds = o.red_odds if picked_red else o.blue_odds
            dec = _american_to_decimal(pick_odds)
            pick_pl = round((dec - 1) * 100 if pick_won else -100, 2)

        # Edge calculation
        edge = None
        edge_side = None
        edge_won = None
        edge_pl = None
        edge_odds_val = None
        if odds_rows:
            avg_red_ip = sum(_implied_prob(o.red_odds) for o in odds_rows) / len(odds_rows)
            avg_blue_ip = sum(_implied_prob(o.blue_odds) for o in odds_rows) / len(odds_rows)

            # Remove the vig before comparing to model probabilities. Raw implied
            # probabilities sum to >1, so edges measured against them are inflated by
            # roughly the book's margin on every fight.
            ip_total = avg_red_ip + avg_blue_ip
            if ip_total > 0:
                avg_red_ip, avg_blue_ip = avg_red_ip / ip_total, avg_blue_ip / ip_total

            model_red = pred.red_prob
            model_blue = 1 - pred.red_prob
            red_edge = model_red - avg_red_ip
            blue_edge = model_blue - avg_blue_ip

            # Price the bet at the SAME book used for the pick P/L. Taking the best
            # line across every book is a price no one could actually have gotten on a
            # historical slate, and it made edge P/L look better than pick P/L for
            # reasons unrelated to the model.
            o = odds_rows[0]
            if red_edge > blue_edge:
                edge_side = "red"
                edge = round(red_edge * 100, 1)
                edge_odds_val = o.red_odds
            else:
                edge_side = "blue"
                edge = round(blue_edge * 100, 1)
                edge_odds_val = o.blue_odds

            edge_won = (edge_side == "red") == red_won
            dec = _american_to_decimal(edge_odds_val)
            edge_pl = round((dec - 1) * 100 if edge_won else -100, 2)

        fight_data.append({
            "conf": round(pred.confidence, 4),
            "edge": edge,
            "pick_won": pick_won,
            "edge_won": edge_won,
            "pick_pl": pick_pl,
            "edge_pl": edge_pl,
        })

    total = len(fight_data)
    correct = sum(1 for f in fight_data if f["pick_won"])

    # Confidence-based splits
    conf_buckets = [
        (0.00, 0.05, "50-55%"),
        (0.05, 0.10, "55-60%"),
        (0.10, 0.15, "60-65%"),
        (0.15, 0.20, "65-70%"),
        (0.20, 0.30, "70-80%"),
        (0.30, 0.50, "80%+"),
    ]
    conf_splits = []
    for lo, hi, label in conf_buckets:
        bucket = [f for f in fight_data if lo <= f["conf"] < hi]
        if not bucket:
            continue
        c = sum(1 for f in bucket if f["pick_won"])
        with_odds = [f for f in bucket if f["pick_pl"] is not None]
        pl = sum(f["pick_pl"] for f in with_odds)
        c_odds = sum(1 for f in with_odds if f["pick_won"])
        # Edge P/L for same confidence bucket
        edge_with_odds = [f for f in bucket if f["edge_pl"] is not None]
        edge_pl = sum(f["edge_pl"] for f in edge_with_odds)
        edge_c = sum(1 for f in edge_with_odds if f["edge_won"])

        conf_splits.append({
            "label": label,
            "fights": len(bucket),
            "correct": c,
            "accuracy": round(c / len(bucket), 4),
            "fights_with_odds": len(with_odds),
            "correct_with_odds": c_odds,
            "accuracy_with_odds": round(c_odds / len(with_odds), 4) if with_odds else 0,
            "pl": round(pl, 2),
            "roi": round(pl / (len(with_odds) * 100) * 100, 2) if with_odds else 0,
            # Edge data for same bucket. The edge-bet count can differ from the pick
            # count, so it is exposed separately rather than reusing fights_with_odds
            # as the ROI denominator.
            "edge_fights_with_odds": len(edge_with_odds),
            "edge_correct": edge_c,
            "edge_accuracy": round(edge_c / len(edge_with_odds), 4) if edge_with_odds else 0,
            "edge_pl": round(edge_pl, 2),
            "edge_roi": round(edge_pl / (len(edge_with_odds) * 100) * 100, 2) if edge_with_odds else 0,
        })

    # Edge-based splits (bucketed by edge percentage)
    edge_buckets = [
        (None, 0, "Negative"),
        (0, 5, "0-5%"),
        (5, 10, "5-10%"),
        (10, 15, "10-15%"),
        (15, 20, "15-20%"),
        (20, 100, "20%+"),
    ]
    edge_splits = []
    for lo, hi, label in edge_buckets:
        if lo is None:
            bucket = [f for f in fight_data if f["edge"] is not None and f["edge"] < 0]
        else:
            bucket = [f for f in fight_data if f["edge"] is not None and lo <= f["edge"] < hi]
        if not bucket:
            continue
        with_odds = [f for f in bucket if f["edge_pl"] is not None]
        edge_c = sum(1 for f in with_odds if f["edge_won"])
        edge_pl = sum(f["edge_pl"] for f in with_odds)
        pick_c = sum(1 for f in with_odds if f["pick_won"])
        pick_pl = sum(f["pick_pl"] for f in with_odds)

        edge_splits.append({
            "label": label,
            "fights": len(bucket),
            "fights_with_odds": len(with_odds),
            "edge_correct": edge_c,
            "edge_accuracy": round(edge_c / len(with_odds), 4) if with_odds else 0,
            "edge_pl": round(edge_pl, 2),
            "edge_roi": round(edge_pl / (len(with_odds) * 100) * 100, 2) if with_odds else 0,
            "pick_correct": pick_c,
            "pick_accuracy": round(pick_c / len(with_odds), 4) if with_odds else 0,
            "pick_pl": round(pick_pl, 2),
            "pick_roi": round(pick_pl / (len(with_odds) * 100) * 100, 2) if with_odds else 0,
        })

    total_with_odds = sum(1 for f in fight_data if f["pick_pl"] is not None)
    total_pl = sum(f["pick_pl"] for f in fight_data if f["pick_pl"] is not None)
    # Count edge bets separately. Dividing edge P/L by the PICK count understates the
    # denominator whenever the two differ, inflating edge ROI.
    edge_bets = [f["edge_pl"] for f in fight_data if f["edge_pl"] is not None]
    total_edge_pl = sum(edge_bets)

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "total_with_odds": total_with_odds,
        "total_pl": round(total_pl, 2),
        "total_roi": round(total_pl / (total_with_odds * 100) * 100, 2) if total_with_odds else 0,
        "total_edge_bets": len(edge_bets),
        "total_edge_pl": round(total_edge_pl, 2),
        "total_edge_roi": round(total_edge_pl / (len(edge_bets) * 100) * 100, 2) if edge_bets else 0,
        "confidence_splits": conf_splits,
        "edge_splits": edge_splits,
        # These metrics are computed over ALL decided fights since 2015, most of which
        # the model trained on, so they are substantially IN-SAMPLE and run far
        # optimistic. The out-of-sample numbers are in models/ufc/h2h/eval_results.json
        # (see `python -m app.services.ufc.model --walk-forward --fresh-glicko`).
        "in_sample": True,
        "out_of_sample_reference": "models/ufc/h2h/eval_results.json",
        "fights": fight_data,
    }


@router.get("/events/{event_id}/predictions")
def get_event_predictions(event_id: int, db: Session = Depends(get_db)):
    """Get model predictions for all fights in an event."""
    fight_ids = [f.id for f in db.query(UFCFight.id).filter(UFCFight.event_id == event_id).all()]
    if not fight_ids:
        return {}
    preds = db.query(UFCFightPrediction).filter(UFCFightPrediction.fight_id.in_(fight_ids)).all()
    return {
        str(p.fight_id): {
            "predicted_winner": p.predicted_winner,
            "confidence": p.confidence,
            "red_prob": p.red_prob,
        }
        for p in preds
    }


@router.get("/events/{event_id}/method-predictions")
def get_event_method_predictions(event_id: int, db: Session = Depends(get_db)):
    """Get method-of-victory predictions for all fights in an event."""
    fight_ids = [f.id for f in db.query(UFCFight.id).filter(UFCFight.event_id == event_id).all()]
    if not fight_ids:
        return {}
    preds = db.query(UFCMethodPrediction).filter(UFCMethodPrediction.fight_id.in_(fight_ids)).all()
    return {
        str(p.fight_id): {
            "predicted_method": p.predicted_method,
            "confidence": p.confidence,
            "ko_prob": p.ko_prob,
            "sub_prob": p.sub_prob,
            "dec_prob": p.dec_prob,
        }
        for p in preds
    }


@router.get("/method/metrics")
def get_method_model_metrics(db: Session = Depends(get_db)):
    """Get method prediction model performance metrics."""
    from sqlalchemy import and_

    rows = (
        db.query(UFCMethodPrediction, UFCFight.method)
        .join(UFCFight, UFCMethodPrediction.fight_id == UFCFight.id)
        .filter(and_(UFCFight.method.isnot(None), UFCFight.date >= "2015-01-01"))
        .all()
    )
    if not rows:
        return {"total": 0, "correct": 0, "accuracy": 0, "per_class": []}

    method_map = {
        "KO/TKO": "KO/TKO", "TKO - Doctor's Stoppage": "KO/TKO", "DQ": "KO/TKO",
        "Submission": "Submission",
        "Decision - Unanimous": "Decision", "Decision - Split": "Decision",
        "Decision - Majority": "Decision", "Decision": "Decision",
    }

    total, correct = 0, 0
    per_class = {c: {"correct": 0, "predicted": 0, "actual": 0} for c in ["KO/TKO", "Submission", "Decision"]}

    for pred, actual_method in rows:
        actual_class = method_map.get(actual_method)
        if actual_class is None:
            continue
        total += 1
        per_class[pred.predicted_method]["predicted"] += 1
        per_class[actual_class]["actual"] += 1
        if pred.predicted_method == actual_class:
            correct += 1
            per_class[actual_class]["correct"] += 1

    class_metrics = []
    for cls_name in ["KO/TKO", "Submission", "Decision"]:
        d = per_class[cls_name]
        precision = d["correct"] / d["predicted"] if d["predicted"] else 0
        recall = d["correct"] / d["actual"] if d["actual"] else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        class_metrics.append({
            "class": cls_name,
            "predicted": d["predicted"],
            "actual": d["actual"],
            "correct": d["correct"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    baseline = max(d["actual"] for d in per_class.values()) / total if total else 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0,
        "baseline_accuracy": round(baseline, 4),
        "per_class": class_metrics,
    }


def _method_label(method: str | None) -> str | None:
    """Normalise a bout's method to the three buckets the UI colours by.

    Mirrors methodLabel() in lib/fighterAnalytics.js, including the order of the tests —
    "TKO - Doctor's Stoppage" has to match KO before anything else gets a look.
    """
    m = (method or "").lower()
    if "ko" in m or "tko" in m:
        return "KO/TKO"
    if "sub" in m:
        return "Submission"
    if "dec" in m:
        return "Decision"
    return method or None


def _recent_form_map(db: Session, fighter_ids: list[int], limit: int = 5) -> dict:
    """Each fighter's last `limit` completed results, most recent first.

    WHY THIS IS SERVED INLINE
    -------------------------
    The Upcoming dashboard draws five W/L/D chips per corner. It used to get them by
    fetching each fighter's ENTIRE bout history from /ufc/fighters/{id}/fights and
    slicing the first five client-side — two requests per fight, ~126 across a full
    slate, to read ten booleans. Those requests dominated the page's prefetch queue and
    were the reason the form chips arrived late when a new fight was selected.

    Five rows per fighter ride along with the card instead, and the chips render from the
    payload with nothing to wait for.

    Draws and no-contests count: they happened, they just have no winner. That matches
    deriveForm, which treats a completed bout as one with a winner OR a method.
    """
    if not fighter_ids:
        return {}
    # Five columns, not whole ORM rows: this sweeps every completed bout of ~25 fighters
    # and hydrating full UFCFight objects for it was the expensive part.
    rows = (
        db.query(
            UFCFight.id,
            UFCFight.red_fighter_id,
            UFCFight.blue_fighter_id,
            UFCFight.winner_id,
            UFCFight.method,
        )
        .filter(or_(
            UFCFight.red_fighter_id.in_(fighter_ids),
            UFCFight.blue_fighter_id.in_(fighter_ids),
        ))
        .filter(or_(UFCFight.winner_id.isnot(None), UFCFight.method.isnot(None)))
        .filter(UFCFight.date.isnot(None))
        .order_by(UFCFight.date.desc())
        .all()
    )
    out: dict = {fid: [] for fid in fighter_ids}
    for fight_id, red_id, blue_id, winner_id, method in rows:
        for fid in (red_id, blue_id):
            bucket = out.get(fid)
            if bucket is None or len(bucket) >= limit:
                continue
            bucket.append({
                "id": str(fight_id),
                "win": winner_id == fid,
                "draw": winner_id is None,
                "method": _method_label(method),
            })
    return out


def _upcoming_corner(fighter, recent_form: list | None = None) -> dict:
    """One fighter as the upcoming card renders them.

    height/reach/trains_at/birthplace are here for the tale of the tape on the Upcoming
    dashboard.
    They are plain columns on the row the query already joinedloads, so serving them adds
    no query — and without them that panel can only ever show age and stance.
    """
    return {
        "id": str(fighter.id),
        "first_name": fighter.first_name,
        "last_name": fighter.last_name,
        "nickname": fighter.nickname,
        "stance": fighter.stance,
        "height": fighter.height,
        "reach": fighter.reach,
        "trains_at": fighter.trains_at,
        "birthplace": fighter.birthplace,
        "wins": fighter.wins,
        "losses": fighter.losses,
        "draws": fighter.draws,
        "country_code": fighter.country_code,
        "image_url": fighter.image_url,
        "recent_form": recent_form or [],
    }


@router.get("/upcoming")
def get_upcoming_events():
    """Get upcoming events with fights and predictions."""
    return response_cache.cached("upcoming", _upcoming_build, UPCOMING_TTL)


def _build_upcoming(db: Session):
    from datetime import date as _date, timedelta

    # Use yesterday as cutoff so events stay visible through the day after (handles UTC offset)
    cutoff = _date.today() - timedelta(days=1)

    events = (
        db.query(UFCEvent)
        .filter(UFCEvent.date >= cutoff)
        .order_by(UFCEvent.date)
        .all()
    )

    # Recent form for every corner on every card, in one query rather than one per event.
    # It is the only lookup here that is not event-scoped — a fighter's last five results
    # have nothing to do with which card they are on — so running it inside the loop cost
    # eight round trips to a remote database for one answer.
    corner_rows = (
        db.query(UFCFight.red_fighter_id, UFCFight.blue_fighter_id)
        .filter(
            UFCFight.event_id.in_([e.id for e in events]),
            UFCFight.winner_id.is_(None),
        )
        .all()
    ) if events else []
    form_map = _recent_form_map(db, list({fid for row in corner_rows for fid in row}))

    result = []
    for event in events:
        fights = (
            db.query(UFCFight)
            .options(
                joinedload(UFCFight.red_fighter),
                joinedload(UFCFight.blue_fighter),
            )
            .filter(UFCFight.event_id == event.id, UFCFight.winner_id.is_(None))
            # Card order as ufcstats lists it: main event first. Bouts scraped before
            # card_position existed sort last rather than jumping to the top.
            .order_by(UFCFight.card_position.is_(None), UFCFight.card_position, UFCFight.id)
            .all()
        )
        if not fights:
            continue

        fight_ids = [f.id for f in fights]
        preds = db.query(UFCFightPrediction).filter(
            UFCFightPrediction.fight_id.in_(fight_ids)
        ).all()
        pred_map = {p.fight_id: p for p in preds}

        method_preds = db.query(UFCMethodPrediction).filter(
            UFCMethodPrediction.fight_id.in_(fight_ids)
        ).all()
        method_pred_map = {mp.fight_id: mp for mp in method_preds}

        odds_rows = db.query(UFCFightOdds).filter(
            UFCFightOdds.fight_id.in_(fight_ids)
        ).all()
        # Group all bookmaker odds per fight
        odds_map = {}
        for o in odds_rows:
            odds_map.setdefault(o.fight_id, []).append(o)

        # Exchange consensus for the whole card in one query — see consensus_probs_bulk.
        from app.services.ufc.prediction_markets.serving import consensus_probs_bulk
        exchange_map = consensus_probs_bulk(db, fight_ids)

        fight_list = []
        for f in fights:
            p = pred_map.get(f.id)
            fight_odds = odds_map.get(f.id, [])
            fight_list.append({
                "id": str(f.id),
                "weight_class": f.weight_class,
                "red_fighter": _upcoming_corner(f.red_fighter, form_map.get(f.red_fighter_id)),
                "blue_fighter": _upcoming_corner(f.blue_fighter, form_map.get(f.blue_fighter_id)),
                "odds": [{
                    "bookmaker": o.bookmaker,
                    "red_odds": o.red_odds,
                    "blue_odds": o.blue_odds,
                    "updated_at": o.updated_at.isoformat() if o.updated_at else None,
                } for o in fight_odds],
                # Volume-weighted Kalshi/Polymarket price for the red corner. Namespaced apart
                # from `odds` because it is a no-vig traded probability, not an American line.
                "exchange": exchange_map.get(f.id),
                "prediction": {
                    "predicted_winner": p.predicted_winner,
                    "confidence": p.confidence,
                    "red_prob": p.red_prob,
                } if p else None,
                "method_prediction": {
                    "predicted_method": mp.predicted_method,
                    "confidence": mp.confidence,
                    "ko_prob": mp.ko_prob,
                    "sub_prob": mp.sub_prob,
                    "dec_prob": mp.dec_prob,
                } if (mp := method_pred_map.get(f.id)) else None,
            })

        result.append({
            # str(): event ids exceed 2**53, so a raw int arrives in JS rounded to the
            # same float for every event on the card and the filter matches all of them.
            "id": str(event.id),
            "name": event.name,
            "date": str(event.date),
            "location": event.location,
            "fights": fight_list,
        })

    return result


@router.get("/rankings")
def get_rankings():
    """Get fighter rankings by weight class (precomputed from full model).

    Each fighter carries a `ledger`: the per-bout decomposition of their score, so the
    ranking can be audited rather than taken on faith.
    """
    return response_cache.cached("rankings", _rankings_build, RANKINGS_TTL)


def _rankings_build():
    # get_rankings opens its own session, so it is already a valid cache thunk.
    from app.services.ufc.tapology_rankings import get_rankings
    return get_rankings()


_upcoming_build = with_session(_build_upcoming)

# The two views behind the landing page and every fighter profile. Warmed at startup
# and after each scheduled job, so the first visitor after a deploy does not pay the
# ~6s cold build.
response_cache.register_warmer("upcoming", _upcoming_build, UPCOMING_TTL)
response_cache.register_warmer("rankings", _rankings_build, RANKINGS_TTL)


@router.get("/arbitrage")
def get_arbitrage_opportunities(db: Session = Depends(get_db)):
    """Find arbitrage opportunities across bookmakers for upcoming fights."""
    return _get_picks_data(db)


@router.get("/picks")
def get_picks(db: Session = Depends(get_db)):
    """Get model picks with edge and arbitrage opportunities for upcoming fights."""
    return _get_picks_data(db)


def _get_picks_data(db: Session):
    from datetime import date as _date, timedelta

    from app.services.ufc.prediction_markets.serving import (
        best_raw_prices as best_exchange_prices, consensus_prob as market_consensus,
    )

    def implied_prob(american_odds):
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    cutoff = _date.today() - timedelta(days=1)

    # Get upcoming fights with multi-book odds
    upcoming_fights = (
        db.query(UFCFight)
        .options(
            joinedload(UFCFight.red_fighter),
            joinedload(UFCFight.blue_fighter),
            joinedload(UFCFight.event),
        )
        .filter(UFCFight.winner_id.is_(None))
        .join(UFCEvent, UFCFight.event_id == UFCEvent.id)
        .filter(UFCEvent.date >= cutoff)
        .all()
    )

    results = []
    for fight in upcoming_fights:
        odds_rows = db.query(UFCFightOdds).filter(UFCFightOdds.fight_id == fight.id).all()
        prediction = db.query(UFCFightPrediction).filter(UFCFightPrediction.fight_id == fight.id).first()
        method_pred = db.query(UFCMethodPrediction).filter(UFCMethodPrediction.fight_id == fight.id).first()

        # Need either odds or prediction to be useful
        if not odds_rows and not prediction:
            continue

        # Arb calculation
        best_red = max(odds_rows, key=lambda o: o.red_odds) if odds_rows else None
        best_blue = max(odds_rows, key=lambda o: o.blue_odds) if odds_rows else None

        arb_margin = None
        is_arb = False
        if best_red and best_blue:
            red_ip = implied_prob(best_red.red_odds)
            blue_ip = implied_prob(best_blue.blue_odds)
            total_ip = red_ip + blue_ip
            arb_margin = round((1 - total_ip) * 100, 2)
            is_arb = total_ip < 1.0

        # Model prediction: who the model thinks wins
        model_winner_side = None
        model_winner_name = None
        model_winner_prob = None
        if prediction:
            model_red_prob = prediction.red_prob
            model_blue_prob = 1 - prediction.red_prob
            if model_red_prob >= 0.5:
                model_winner_side = "red"
                model_winner_name = f"{fight.red_fighter.first_name} {fight.red_fighter.last_name}"
                model_winner_prob = round(model_red_prob * 100, 1)
            else:
                model_winner_side = "blue"
                model_winner_name = f"{fight.blue_fighter.first_name} {fight.blue_fighter.last_name}"
                model_winner_prob = round(model_blue_prob * 100, 1)

        # Edge pick: which side has the best betting value (can differ from model winner)
        edge = None
        edge_side = None
        edge_fighter = None
        edge_model_prob = None
        edge_implied_prob = None
        if prediction and odds_rows:
            # Use average implied prob across all books as consensus
            avg_red_ip = sum(implied_prob(o.red_odds) for o in odds_rows) / len(odds_rows)
            avg_blue_ip = sum(implied_prob(o.blue_odds) for o in odds_rows) / len(odds_rows)

            model_red_prob = prediction.red_prob
            model_blue_prob = 1 - prediction.red_prob

            # Edge = model probability - market implied probability
            red_edge = model_red_prob - avg_red_ip
            blue_edge = model_blue_prob - avg_blue_ip

            # Pick the side with the bigger edge (best value bet)
            if red_edge > blue_edge:
                edge_side = "red"
                edge_fighter = f"{fight.red_fighter.first_name} {fight.red_fighter.last_name}"
                edge = round(red_edge * 100, 1)
                edge_model_prob = round(model_red_prob * 100, 1)
                edge_implied_prob = round(avg_red_ip * 100, 1)
            else:
                edge_side = "blue"
                edge_fighter = f"{fight.blue_fighter.first_name} {fight.blue_fighter.last_name}"
                edge = round(blue_edge * 100, 1)
                edge_model_prob = round(model_blue_prob * 100, 1)
                edge_implied_prob = round(avg_blue_ip * 100, 1)

        all_books = [{
            "bookmaker": o.bookmaker,
            "red_odds": o.red_odds,
            "blue_odds": o.blue_odds,
        } for o in sorted(odds_rows, key=lambda o: o.bookmaker)]

        # Exchange pricing, kept in its own fields rather than mixed into `odds_rows`.
        #
        # This endpoint powers the site's picks/arbitrage page. The *pre-registered* rule lives in
        # app/services/ufc/picks.py and scripts/generate_picks.py, and PREREGISTRATION.md registers
        # its book set; letting exchange prices into the average above would change the registered
        # market price and silently invalidate the rule. They are separate code paths and the
        # exchange data is in separate tables, so that cannot happen by accident -- but the numbers
        # are kept separate here too, so the page can show both without either being mistaken for
        # the other.
        exchange = market_consensus(db, fight.id)
        exchange_edge = None
        if exchange and prediction:
            # Against a no-vig exchange price there is no devig assumption in the comparison,
            # which makes this the more honest of the two edges shown.
            side_prob = (prediction.red_prob if (edge_side or "red") == "red"
                         else 1 - prediction.red_prob)
            market_side = (exchange["red_prob"] if (edge_side or "red") == "red"
                           else 1 - exchange["red_prob"])
            exchange_edge = round((side_prob - market_side) * 100, 1)

        # Cross-venue arbitrage: an exchange price against a sportsbook price. Because exchange
        # quotes carry no vig, this pairing is far likelier to cross than book-vs-book, which is
        # what the existing `margin` measures.
        cross_margin, cross_detail = None, None
        raw = best_exchange_prices(db, fight.id)
        if raw and best_red and best_blue:
            # Raw exchange prices, not the normalised consensus: normalising forces red + blue to
            # exactly 1 and so guarantees a margin of zero, which is the one answer an arbitrage
            # check must never be hard-coded to give.
            book_red, book_blue = implied_prob(best_red.red_odds), implied_prob(best_blue.blue_odds)
            red_src = "exchange" if raw["red_price"] < book_red else "book"
            blue_src = "exchange" if raw["blue_price"] < book_blue else "book"
            total = min(raw["red_price"], book_red) + min(raw["blue_price"], book_blue)
            cross_margin = round((1 - total) * 100, 2)
            cross_detail = {
                "red_source": raw["red_venue"] if red_src == "exchange" else best_red.bookmaker,
                "blue_source": raw["blue_venue"] if blue_src == "exchange" else best_blue.bookmaker,
            }

        results.append({
            "fight_id": fight.id,
            "event_name": fight.event.name if fight.event else "",
            "event_date": str(fight.event.date) if fight.event else "",
            "weight_class": fight.weight_class,
            "red_fighter": f"{fight.red_fighter.first_name} {fight.red_fighter.last_name}",
            "blue_fighter": f"{fight.blue_fighter.first_name} {fight.blue_fighter.last_name}",
            # Arb data
            "best_red_odds": best_red.red_odds if best_red else None,
            "best_red_book": best_red.bookmaker if best_red else None,
            "best_blue_odds": best_blue.blue_odds if best_blue else None,
            "best_blue_book": best_blue.bookmaker if best_blue else None,
            "margin": arb_margin,
            "is_arb": is_arb,
            # Model prediction (who the model thinks wins)
            "model_winner_side": model_winner_side,
            "model_winner_name": model_winner_name,
            "model_winner_prob": model_winner_prob,
            "confidence": round(prediction.confidence * 100, 1) if prediction else None,
            "method_prediction": method_pred.predicted_method if method_pred else None,
            # Edge pick (best value bet - can differ from model winner)
            "edge_side": edge_side,
            "edge_fighter": edge_fighter,
            "edge": edge,
            "edge_model_prob": edge_model_prob,
            "edge_implied_prob": edge_implied_prob,
            # Odds
            "all_odds": all_books,
            # Prediction markets, deliberately namespaced apart from the sportsbook fields above.
            "exchange_red_prob": round(exchange["red_prob"] * 100, 1) if exchange else None,
            "exchange_venues": exchange["venues"] if exchange else None,
            "exchange_edge": exchange_edge,
            "cross_venue_margin": cross_margin,
            "cross_venue_detail": cross_detail,
            "updated_at": max(
                (o.updated_at for o in odds_rows if o.updated_at),
                default=None,
            ),
        })

    # Sort by edge (biggest model edge first), then arb margin
    results.sort(key=lambda x: -(x["edge"] or -999))

    for r in results:
        if r["updated_at"]:
            r["updated_at"] = r["updated_at"].isoformat()

    return results
