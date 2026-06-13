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
    """Parse scheduled rounds from time_format (e.g. '3 Rnd (5-5-5)')."""
    fmt = fight.time_format
    if not fmt:
        return None
    parts = fmt.split()
    if parts and parts[0].isdigit():
        return int(parts[0])
    return None


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
            "finish_rates": _finish_rates(db, f.id),
            "days_since_last_fight": _days_since_last_fight(db, f.id),
            "division_info": _division_change(db, f.id, fight.weight_class),
        }

    return {
        "event": {
            "name": event.name if event else "Unknown",
            "date": str(event.date) if event else "Unknown",
            "location": event.location if event else None,
        },
        "weight_class": fight.weight_class,
        "scheduled_rounds": _scheduled_rounds(fight),
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

Write an analytical preview in markdown. Be specific with data. Reference actual stats, records, and numbers from the data provided. Do not use generic filler.

Rules:
- Use the model feature data to inform your analysis, but never say "SHAP", "SHAP value", "feature importance", or reference model internals. Present insights as your own fight analysis grounded in the stats (e.g. instead of "the age difference SHAP value of -0.3016 suggests..." say "At 38, Pereira's age could be a factor as he moves up to heavyweight").
- Use bold sparingly, only for section headers. Do not bold phrases or words within paragraphs, except for the final prediction sentence in the last section.
- Never use em dashes (the long dash). Use commas, periods, or semicolons instead.
- When referring to time since last fight, convert days into natural units: use "X months" for 30+ days, "X weeks" for 7-29 days, "X days" only for less than a week.
- You may include markdown tables anywhere they help illustrate a point (striking comparisons, recent results, tale of the tape, etc.). This is optional and up to your judgment. Pull numbers directly from the data provided.

Structure every preview with exactly these sections:

# [Red Last Name] vs. [Blue Last Name] | [Event Name]: [catchy one-liner]
The one-liner should be part of the title itself after a colon. Keep it short and punchy.

## Overview
2-3 sentences setting up the matchup.

## Tale of the Tape
A markdown table comparing key attributes: record, age, height, reach, stance.

## [Red Fighter Name]
Analysis of recent form using their last few fights. You can include a table of recent results if it supports your point.

## [Blue Fighter Name]
Same format as above.

## Key Factors
Write this as flowing prose, not a bulleted list. Weave 3-5 factors together into a cohesive paragraph or two that tells the story of how this fight will be decided. Ground these in the data: striking rates, takedown numbers, finish rates, age, reach advantages, quality of opposition. Use the model's top features as a guide but describe them as fight dynamics, not model outputs.

## How This Fight Plays Out
A short narrative (3-4 sentences) describing how you see the fight unfolding. Include how it compares to the betting odds and note any value gaps between the model and the market.

**Prediction: [Fighter Last Name] by [Method].**
This line must appear as its own paragraph at the very end, fully bolded. It is not a section header.

Keep the total length to about 600-800 words."""

    import json

    def _fighter_block(key: str, context: dict) -> str:
        f = context[f'{key}_fighter']
        fr = f['finish_rates']
        di = f['division_info']
        days = f['days_since_last_fight']
        lines = [
            f"**{key.upper()} CORNER:** {f['name']}",
            f"- Record: {f['record']}",
            f"- Height: {f['height'] or 'N/A'}, Reach: {f['reach'] or 'N/A'}",
            f"- Stance: {f['stance'] or 'N/A'}, Age: {f['age'] or 'N/A'}",
            f"- Nickname: {f['nickname'] or 'None'}",
            f"- Finish rates: {int(fr['ko_rate']*100)}% KO, {int(fr['sub_rate']*100)}% SUB ({fr['total_wins']} UFC wins)",
            f"- Days since last fight: {days if days is not None else 'N/A'}",
        ]
        if di['ufc_debut']:
            lines.append("- UFC DEBUT")
        elif di['division_change']:
            lines.append(f"- DIVISION CHANGE (previously {di['previous_division']})")
        lines.append("")
        lines.append("Recent Fights:")
        lines.append(json.dumps(context[f'{key}_recent_fights'], indent=2))
        return "\n".join(lines)

    user = f"""Generate a fight preview for the following upcoming bout.

**Event:** {context['event']['name']}, {context['event']['date']}
**Weight Class:** {context['weight_class'] or 'Unknown'}
**Scheduled Rounds:** {context.get('scheduled_rounds') or 'Unknown'}

{_fighter_block('red', context)}

{_fighter_block('blue', context)}

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
