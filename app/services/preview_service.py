"""
AI-powered fight preview generation using Claude API.

Gathers fighter data, predictions, SHAP values, and odds to generate
rich markdown previews for upcoming UFC fights.
"""

import logging
import time

import anthropic
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models.ufc import (
    UFCEvent, UFCFight, UFCFighter, UFCFightOdds,
    UFCFightPrediction, UFCFightPreview, UFCFightShapValue,
    UFCFightStats, UFCMethodPrediction,
)

log = logging.getLogger(__name__)


def _format_ctrl(seconds: int) -> str:
    if not seconds:
        return "0:00"
    return f"{seconds // 60}:{seconds % 60:02d}"


def _format_odds(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)


def _fighter_recent_fights(db: Session, fighter_id: int, limit: int = 5) -> list[dict]:
    """Get a fighter's most recent fights with stats."""
    fights = (
        db.query(UFCFight)
        .filter(
            or_(UFCFight.red_fighter_id == fighter_id, UFCFight.blue_fighter_id == fighter_id),
            UFCFight.winner_id.isnot(None),
        )
        .order_by(UFCFight.date.desc())
        .limit(limit)
        .all()
    )

    results = []
    for fight in fights:
        corner = "red" if fight.red_fighter_id == fighter_id else "blue"
        opp_id = fight.blue_fighter_id if corner == "red" else fight.red_fighter_id
        opponent = db.query(UFCFighter).filter(UFCFighter.id == opp_id).first()
        won = fight.winner_id == fighter_id

        # Get totals row
        totals = (
            db.query(UFCFightStats)
            .filter(
                UFCFightStats.fight_id == fight.id,
                UFCFightStats.fighter_id == fighter_id,
                UFCFightStats.round_number == 0,
            )
            .first()
        )

        fight_info = {
            "date": str(fight.date) if fight.date else "Unknown",
            "opponent": f"{opponent.first_name} {opponent.last_name}" if opponent else "Unknown",
            "result": "Win" if won else "Loss",
            "method": fight.method or "Unknown",
            "round": fight.finish_round,
            "weight_class": fight.weight_class,
        }

        if totals:
            fight_info["stats"] = {
                "sig_str": f"{totals.sig_str_landed}/{totals.sig_str_attempted}",
                "td": f"{totals.td_landed}/{totals.td_attempted}",
                "kd": totals.kd,
                "sub_att": totals.sub_att,
                "ctrl": _format_ctrl(totals.ctrl_seconds),
            }

        results.append(fight_info)

    return results


def gather_fight_context(fight_id: int, db: Session) -> dict | None:
    """Collect all data needed for a fight preview."""
    fight = db.query(UFCFight).filter(UFCFight.id == fight_id).first()
    if not fight:
        return None

    red = db.query(UFCFighter).filter(UFCFighter.id == fight.red_fighter_id).first()
    blue = db.query(UFCFighter).filter(UFCFighter.id == fight.blue_fighter_id).first()
    if not red or not blue:
        return None

    event = db.query(UFCEvent).filter(UFCEvent.id == fight.event_id).first()

    prediction = db.query(UFCFightPrediction).filter(UFCFightPrediction.fight_id == fight_id).first()
    method_pred = db.query(UFCMethodPrediction).filter(UFCMethodPrediction.fight_id == fight_id).first()

    shap_rows = (
        db.query(UFCFightShapValue)
        .filter(UFCFightShapValue.fight_id == fight_id)
        .order_by(UFCFightShapValue.abs_value.desc())
        .limit(10)
        .all()
    )

    odds_rows = db.query(UFCFightOdds).filter(UFCFightOdds.fight_id == fight_id).all()

    def fighter_dict(f):
        from datetime import date
        age = None
        if f.dob:
            try:
                today = date.today()
                dob = f.dob if isinstance(f.dob, date) else date.fromisoformat(str(f.dob))
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            except Exception:
                pass
        return {
            "name": f"{f.first_name} {f.last_name}",
            "nickname": f.nickname,
            "record": f"{f.wins}-{f.losses}-{f.draws}",
            "height": f.height,
            "weight": f.weight,
            "reach": f.reach,
            "stance": f.stance,
            "age": age,
        }

    return {
        "event": {
            "name": event.name if event else "Unknown",
            "date": str(event.date) if event else "Unknown",
            "location": event.location if event else None,
        },
        "weight_class": fight.weight_class,
        "red_fighter": fighter_dict(red),
        "red_recent_fights": _fighter_recent_fights(db, red.id),
        "blue_fighter": fighter_dict(blue),
        "blue_recent_fights": _fighter_recent_fights(db, blue.id),
        "prediction": {
            "predicted_winner": prediction.predicted_winner,
            "red_prob": round(prediction.red_prob, 3),
            "confidence": round(prediction.confidence, 3),
        } if prediction else None,
        "method_prediction": {
            "predicted_method": method_pred.predicted_method,
            "ko_prob": round(method_pred.ko_prob, 3),
            "sub_prob": round(method_pred.sub_prob, 3),
            "dec_prob": round(method_pred.dec_prob, 3),
        } if method_pred else None,
        "shap_values": [
            {
                "feature": s.feature_name,
                "value": round(s.shap_value, 4),
                "feature_value": round(s.feature_value, 3) if s.feature_value is not None else None,
            }
            for s in shap_rows
        ],
        "odds": [
            {
                "bookmaker": o.bookmaker,
                "red_odds": _format_odds(o.red_odds),
                "blue_odds": _format_odds(o.blue_odds),
            }
            for o in odds_rows
        ],
    }


def build_preview_prompt(context: dict) -> tuple[str, str]:
    """Build system and user prompts for Claude API."""
    system = """You are an expert MMA analyst writing a pre-fight preview for a sports analytics platform.

Write an engaging, analytical preview in markdown. Be specific with data — reference actual stats, records, and numbers. Do not use generic filler.

Structure your preview with these sections:
1. **Overview** — A 2-3 sentence hook about the matchup
2. **Fighter Comparison** — A markdown table comparing key attributes (record, height, reach, stance, age, streak)
3. **Red Corner: [Name]** — Analysis of their recent form, strengths, and tendencies (reference their last few fights)
4. **Blue Corner: [Name]** — Same analysis for the other fighter
5. **Key Matchup Factors** — Interpret the model's top features (SHAP values) in plain English. Explain what each factor means for how the fight plays out
6. **Model Prediction** — Present the model's pick, confidence, method prediction, and how it compares to the betting odds. Note any value discrepancies
7. **How This Fight Plays Out** — Your narrative prediction of how the fight unfolds

Keep the total length to about 600-800 words. Use bold for emphasis. SHAP feature names use underscores — translate them to readable English (e.g. "diff_avg_sig_str_landed_per5" → "significant striking rate advantage")."""

    import json
    user = f"""Generate a fight preview for the following upcoming bout.

**Event:** {context['event']['name']} — {context['event']['date']}
**Weight Class:** {context['weight_class'] or 'Unknown'}

**RED CORNER:** {context['red_fighter']['name']}
- Record: {context['red_fighter']['record']}
- Height: {context['red_fighter']['height'] or 'N/A'}, Reach: {context['red_fighter']['reach'] or 'N/A'}
- Stance: {context['red_fighter']['stance'] or 'N/A'}, Age: {context['red_fighter']['age'] or 'N/A'}
- Nickname: {context['red_fighter']['nickname'] or 'None'}

Recent Fights:
{json.dumps(context['red_recent_fights'], indent=2)}

**BLUE CORNER:** {context['blue_fighter']['name']}
- Record: {context['blue_fighter']['record']}
- Height: {context['blue_fighter']['height'] or 'N/A'}, Reach: {context['blue_fighter']['reach'] or 'N/A'}
- Stance: {context['blue_fighter']['stance'] or 'N/A'}, Age: {context['blue_fighter']['age'] or 'N/A'}
- Nickname: {context['blue_fighter']['nickname'] or 'None'}

Recent Fights:
{json.dumps(context['blue_recent_fights'], indent=2)}

**MODEL PREDICTION:**
{json.dumps(context['prediction'], indent=2) if context['prediction'] else 'No prediction available'}

**METHOD PREDICTION:**
{json.dumps(context['method_prediction'], indent=2) if context['method_prediction'] else 'No method prediction available'}

**TOP SHAP FEATURES (positive = favors red, negative = favors blue):**
{json.dumps(context['shap_values'], indent=2) if context['shap_values'] else 'No SHAP values available'}

**ODDS:**
{json.dumps(context['odds'], indent=2) if context['odds'] else 'No odds available'}"""

    return system, user


def generate_preview(fight_id: int, db: Session, force: bool = False) -> UFCFightPreview | None:
    """Generate an AI preview for a single fight."""
    if not settings.ANTHROPIC_API_KEY:
        log.warning("ANTHROPIC_API_KEY not set, skipping preview generation")
        return None

    # Check existing
    existing = db.query(UFCFightPreview).filter(UFCFightPreview.fight_id == fight_id).first()
    if existing and not force:
        return existing

    context = gather_fight_context(fight_id, db)
    if not context:
        log.warning(f"Could not gather context for fight {fight_id}")
        return None

    system_prompt, user_message = build_preview_prompt(context)

    try:
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        content = response.content[0].text
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

        if existing:
            existing.content = content
            existing.model_used = settings.ANTHROPIC_MODEL
            existing.prompt_tokens = prompt_tokens
            existing.completion_tokens = completion_tokens
        else:
            existing = UFCFightPreview(
                fight_id=fight_id,
                content=content,
                model_used=settings.ANTHROPIC_MODEL,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            db.add(existing)

        db.commit()
        log.info(f"Generated preview for fight {fight_id} ({prompt_tokens}+{completion_tokens} tokens)")
        return existing

    except Exception as e:
        log.error(f"Failed to generate preview for fight {fight_id}: {e}")
        return None


def generate_all_upcoming_previews(force: bool = False):
    """Generate previews for all upcoming fights that have predictions."""
    from datetime import date

    db = SessionLocal()
    try:
        # Collect IDs first, then close the query session to avoid transaction conflicts
        fight_ids = [
            row[0] for row in
            db.query(UFCFight.id)
            .join(UFCEvent, UFCFight.event_id == UFCEvent.id)
            .join(UFCFightPrediction, UFCFightPrediction.fight_id == UFCFight.id)
            .filter(
                UFCFight.winner_id.is_(None),
                UFCEvent.date >= date.today(),
            )
            .all()
        ]
        db.close()

        log.info(f"Generating previews for {len(fight_ids)} upcoming fights")

        for fight_id in fight_ids:
            session = SessionLocal()
            try:
                generate_preview(fight_id, session, force=force)
            except Exception:
                log.exception(f"Failed to generate preview for fight {fight_id}")
            finally:
                session.close()
            time.sleep(1)  # Rate limit courtesy

        log.info("Finished generating all upcoming previews")

    except Exception:
        log.exception("Failed to generate upcoming previews")
    finally:
        try:
            db.close()
        except Exception:
            pass
