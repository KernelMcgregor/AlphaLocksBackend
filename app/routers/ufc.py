from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models.ufc import (
    UFCEvent, UFCFight, UFCFighter, UFCFightOdds,
    UFCFightPrediction, UFCFightPreview, UFCMethodPrediction, UFCFightShapValue, UFCFightStats,
)
from app.schemas.ufc import (
    UFCEventDetailResponse,
    UFCEventResponse,
    UFCFightDetailResponse,
    UFCFightResponse,
    UFCFighterResponse,
    UFCFightStatsResponse,
)

router = APIRouter(prefix="/ufc", tags=["ufc"])


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
def get_fight(fight_id: int, db: Session = Depends(get_db)):
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
    } for o in odds_rows]
    result["shap_values"] = [{
        "feature_name": s.feature_name,
        "shap_value": s.shap_value,
        "abs_value": s.abs_value,
        "feature_value": s.feature_value,
    } for s in shap_rows]

    # Add preview
    preview = db.query(UFCFightPreview).filter(UFCFightPreview.fight_id == fight_id).first()
    result["preview"] = {
        "content": preview.content,
        "model_used": preview.model_used,
        "generated_at": preview.created_at.isoformat() if preview.created_at else None,
    } if preview else None

    return result


# --- Predictions ---

@router.get("/model/metrics")
def get_model_metrics(db: Session = Depends(get_db)):
    """Get model performance metrics with real odds P/L."""
    from sqlalchemy import and_
    from sqlalchemy.orm import aliased

    rows = (
        db.query(UFCFightPrediction, UFCFight.winner_id, UFCFight.red_fighter_id, UFCFightOdds)
        .join(UFCFight, UFCFightPrediction.fight_id == UFCFight.id)
        .outerjoin(UFCFightOdds, UFCFightOdds.fight_id == UFCFightPrediction.fight_id)
        .filter(and_(UFCFight.winner_id.isnot(None), UFCFight.date >= "2015-01-01"))
        .all()
    )
    if not rows:
        return {"total": 0, "correct": 0, "accuracy": 0, "confidence_splits": []}

    def _american_to_decimal(odds):
        if odds > 0:
            return 1 + odds / 100
        return 1 + 100 / abs(odds)

    total = len(rows)
    correct = sum(1 for p, w, r, o in rows if (p.predicted_winner == "red") == (w == r))

    buckets = [
        (0.00, 0.05, "50-55%"),
        (0.05, 0.10, "55-60%"),
        (0.10, 0.15, "60-65%"),
        (0.15, 0.20, "65-70%"),
        (0.20, 0.30, "70-80%"),
        (0.30, 0.50, "80%+"),
    ]
    splits = []
    for lo, hi, label in buckets:
        bucket = [(p, w, r, o) for p, w, r, o in rows if lo <= p.confidence < hi]
        if not bucket:
            continue
        c = sum(1 for p, w, r, o in bucket if (p.predicted_winner == "red") == (w == r))

        # P/L calculation: only fights with real odds
        pl = 0.0
        fights_with_odds = 0
        correct_with_odds = 0
        for p, w, r, o in bucket:
            if not (o and o.red_odds and o.blue_odds):
                continue
            fights_with_odds += 1
            picked_red = p.predicted_winner == "red"
            won = picked_red == (w == r)
            if won:
                correct_with_odds += 1
            dec_odds = _american_to_decimal(o.red_odds if picked_red else o.blue_odds)
            pl += (dec_odds - 1) * 100 if won else -100

        splits.append({
            "label": label,
            "fights": len(bucket),
            "correct": c,
            "accuracy": round(c / len(bucket), 4),
            "fights_with_odds": fights_with_odds,
            "correct_with_odds": correct_with_odds,
            "accuracy_with_odds": round(correct_with_odds / fights_with_odds, 4) if fights_with_odds else 0,
            "pl": round(pl, 2),
            "roi": round(pl / (fights_with_odds * 100) * 100, 2) if fights_with_odds else 0,
        })

    total_odds_fights = sum(s["fights_with_odds"] for s in splits)
    total_odds_correct = sum(s["correct_with_odds"] for s in splits)
    total_pl = sum(s["pl"] for s in splits)

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "total_with_odds": total_odds_fights,
        "total_pl": round(total_pl, 2),
        "total_roi": round(total_pl / (total_odds_fights * 100) * 100, 2) if total_odds_fights else 0,
        "confidence_splits": splits,
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


@router.get("/upcoming")
def get_upcoming_events(db: Session = Depends(get_db)):
    """Get upcoming events with fights and predictions."""
    from datetime import date as _date

    events = (
        db.query(UFCEvent)
        .filter(UFCEvent.date >= _date.today())
        .order_by(UFCEvent.date)
        .all()
    )

    result = []
    for event in events:
        fights = (
            db.query(UFCFight)
            .options(
                joinedload(UFCFight.red_fighter),
                joinedload(UFCFight.blue_fighter),
            )
            .filter(UFCFight.event_id == event.id, UFCFight.winner_id.is_(None))
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

        fight_list = []
        for f in fights:
            p = pred_map.get(f.id)
            fight_odds = odds_map.get(f.id, [])
            fight_list.append({
                "id": str(f.id),
                "weight_class": f.weight_class,
                "red_fighter": {
                    "id": str(f.red_fighter.id),
                    "first_name": f.red_fighter.first_name,
                    "last_name": f.red_fighter.last_name,
                    "nickname": f.red_fighter.nickname,
                    "stance": f.red_fighter.stance,
                    "wins": f.red_fighter.wins,
                    "losses": f.red_fighter.losses,
                    "draws": f.red_fighter.draws,
                    "country_code": f.red_fighter.country_code,
                },
                "blue_fighter": {
                    "id": str(f.blue_fighter.id),
                    "first_name": f.blue_fighter.first_name,
                    "last_name": f.blue_fighter.last_name,
                    "nickname": f.blue_fighter.nickname,
                    "stance": f.blue_fighter.stance,
                    "wins": f.blue_fighter.wins,
                    "losses": f.blue_fighter.losses,
                    "draws": f.blue_fighter.draws,
                    "country_code": f.blue_fighter.country_code,
                },
                "odds": [{
                    "bookmaker": o.bookmaker,
                    "red_odds": o.red_odds,
                    "blue_odds": o.blue_odds,
                    "updated_at": o.updated_at.isoformat() if o.updated_at else None,
                } for o in fight_odds],
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
            "id": event.id,
            "name": event.name,
            "date": str(event.date),
            "location": event.location,
            "fights": fight_list,
        })

    return result


@router.get("/arbitrage")
def get_arbitrage_opportunities(db: Session = Depends(get_db)):
    """Find arbitrage opportunities across bookmakers for upcoming fights."""
    return _get_picks_data(db)


@router.get("/picks")
def get_picks(db: Session = Depends(get_db)):
    """Get model picks with edge and arbitrage opportunities for upcoming fights."""
    return _get_picks_data(db)


def _get_picks_data(db: Session):
    from datetime import date as _date

    def implied_prob(american_odds):
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

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
        .filter(UFCEvent.date >= _date.today())
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

        # Model edge: compare model probability to consensus implied probability
        edge = None
        pick_side = None
        pick_fighter = None
        model_prob = None
        implied_prob_pick = None
        if prediction and odds_rows:
            # Use average implied prob across all books as consensus
            avg_red_ip = sum(implied_prob(o.red_odds) for o in odds_rows) / len(odds_rows)
            avg_blue_ip = sum(implied_prob(o.blue_odds) for o in odds_rows) / len(odds_rows)

            model_red_prob = prediction.red_prob
            model_blue_prob = 1 - prediction.red_prob

            # Edge = model probability - market implied probability
            red_edge = model_red_prob - avg_red_ip
            blue_edge = model_blue_prob - avg_blue_ip

            # Pick the side with the bigger edge
            if red_edge > blue_edge:
                pick_side = "red"
                pick_fighter = f"{fight.red_fighter.first_name} {fight.red_fighter.last_name}"
                edge = round(red_edge * 100, 1)
                model_prob = round(model_red_prob * 100, 1)
                implied_prob_pick = round(avg_red_ip * 100, 1)
            else:
                pick_side = "blue"
                pick_fighter = f"{fight.blue_fighter.first_name} {fight.blue_fighter.last_name}"
                edge = round(blue_edge * 100, 1)
                model_prob = round(model_blue_prob * 100, 1)
                implied_prob_pick = round(avg_blue_ip * 100, 1)

        all_books = [{
            "bookmaker": o.bookmaker,
            "red_odds": o.red_odds,
            "blue_odds": o.blue_odds,
        } for o in sorted(odds_rows, key=lambda o: o.bookmaker)]

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
            # Model edge data
            "pick_side": pick_side,
            "pick_fighter": pick_fighter,
            "edge": edge,
            "model_prob": model_prob,
            "implied_prob": implied_prob_pick,
            "confidence": round(prediction.confidence * 100, 1) if prediction else None,
            "method_prediction": method_pred.predicted_method if method_pred else None,
            # Odds
            "all_odds": all_books,
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
