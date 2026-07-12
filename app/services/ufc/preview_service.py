"""
AI-powered fight preview generation using DeepSeek API (OpenAI-compatible).

Gathers fighter data, Glicko component ratings, predictions, SHAP values,
and odds to generate rich markdown previews for upcoming UFC fights.
"""

import logging
import time

from openai import OpenAI
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models.ufc import (
    UFCEvent, UFCFight, UFCFighter, UFCFightOdds,
    UFCFightPrediction, UFCFightPreview, UFCFightShapValue,
    UFCFightStats, UFCGlickoSnapshot, UFCFighterRanking,
    UFCMethodPrediction,
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
    if not snapshot:
        return None

    ranking = (
        db.query(UFCFighterRanking)
        .filter(UFCFighterRanking.fighter_id == fighter_id)
        .first()
    ) if weight_class else None

    # Parse percentile profile from ranking
    import json
    percentiles = {}
    if ranking and ranking.feature_profile:
        try:
            profile = json.loads(ranking.feature_profile)
            percentiles = {k: v.get("percentile", 50) for k, v in profile.items() if isinstance(v, dict) and "percentile" in v}
        except (json.JSONDecodeError, AttributeError):
            pass

    ratings = {}
    for dim in GLICKO_DIMS:
        val = getattr(snapshot, dim, None)
        pct = percentiles.get(dim)
        ratings[dim] = {
            "label": GLICKO_LABELS[dim],
            "rating": round(val, 1) if val is not None else None,
            "percentile": round(pct, 1) if pct is not None else None,
            "tier": _percentile_tier(pct) if pct is not None else None,
        }

    result = {"dimensions": ratings}
    if ranking:
        result["division_rank"] = ranking.rank
        result["expected_win_rate"] = round(ranking.score * 100, 1)

    return result


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
        "red_glicko": _get_glicko_data(db, fight_id, red.id, fight.weight_class),
        "blue_glicko": _get_glicko_data(db, fight_id, blue.id, fight.weight_class),
    }


def _glicko_block(label: str, glicko: dict | None) -> str:
    """Format Glicko data for the prompt."""
    if not glicko:
        return f"**{label} GLICKO RATINGS:** Not available"

    lines = [f"**{label} GLICKO COMPONENT RATINGS:**"]

    if "division_rank" in glicko:
        lines.append(f"- Division Rank: #{glicko['division_rank']}, Expected Win Rate: {glicko['expected_win_rate']}%")

    dims = glicko["dimensions"]
    groups = {
        "Striking": ["str_vol", "str_acc", "str_def", "dist", "clinch"],
        "Power & Durability": ["ko", "kod", "durability"],
        "Grappling": ["td", "tdd", "ctrl", "sub", "subd", "gnd"],
        "Overall": ["pts"],
    }
    for group_name, keys in groups.items():
        parts = []
        for k in keys:
            d = dims.get(k)
            if d and d["percentile"] is not None:
                parts.append(f"{d['label']}: {d['tier']} ({d['percentile']:.0f}th pct, rating {d['rating']})")
        if parts:
            lines.append(f"- {group_name}: {'; '.join(parts)}")

    return "\n".join(lines)


def _glicko_matchup_edges(red_glicko: dict | None, blue_glicko: dict | None) -> str:
    """Compute biggest Glicko differentials between fighters."""
    if not red_glicko or not blue_glicko:
        return ""

    edges = []
    for dim in GLICKO_DIMS:
        rd = red_glicko["dimensions"].get(dim, {})
        bd = blue_glicko["dimensions"].get(dim, {})
        if rd.get("rating") is not None and bd.get("rating") is not None:
            diff = rd["rating"] - bd["rating"]
            r_pct = rd.get("percentile", "?")
            b_pct = bd.get("percentile", "?")
            edges.append((abs(diff), dim, diff, r_pct, b_pct))

    edges.sort(reverse=True)
    if not edges:
        return ""

    lines = ["**KEY MATCHUP EDGES (biggest Glicko differentials):**"]
    for _, dim, diff, r_pct, b_pct in edges[:6]:
        label = GLICKO_LABELS[dim]
        if diff > 0:
            lines.append(f"- Red {label} advantage: +{diff:.0f} rating pts ({r_pct:.0f}th vs {b_pct:.0f}th percentile)")
        else:
            lines.append(f"- Blue {label} advantage: +{abs(diff):.0f} rating pts ({b_pct:.0f}th vs {r_pct:.0f}th percentile)")

    return "\n".join(lines)


def build_preview_prompt(context: dict) -> tuple[str, str]:
    """Build system and user prompts for the preview LLM."""
    system = """You are an expert MMA analyst writing a pre-fight preview for a sports analytics platform.

Write an analytical preview in markdown. Be specific with data. Reference actual stats, records, and numbers from the data provided. Do not use generic filler.

Rules:
- Use the Glicko component ratings to identify each fighter's strengths and weaknesses across 15 skill dimensions. Reference percentile rankings and tiers (Elite, Strong, Average, Below Avg, Weak) to quantify skill edges. Always back up a Glicko insight with concrete fight stats from the recent fights data. For example: "Fighter A's Elite striking defense (94th percentile) is reflected in his 68% significant strike defense across his last 5 fights."
- Use the model feature data to inform your analysis, but never say "SHAP", "SHAP value", "feature importance", "Glicko", "rating system", "component rating", or reference model internals. Present insights as your own fight analysis grounded in the stats. Translate Glicko tiers into natural analyst language (e.g. "elite-level takedown defense", "one of the division's best chins", "above-average submission threat").
- Use bold sparingly, only for section headers. Do not bold phrases or words within paragraphs, except for the final prediction sentence in the last section.
- Never use em dashes (the long dash). Use commas, periods, or semicolons instead.
- When referring to time since last fight, convert days into natural units: use "X months" for 30+ days, "X weeks" for 7-29 days, "X days" only for less than a week.
- You may include markdown tables anywhere they help illustrate a point (striking comparisons, recent results, tale of the tape, etc.). This is optional and up to your judgment. Pull numbers directly from the data provided.

Structure every preview with exactly these sections:

# [Red Last Name] vs. [Blue Last Name] | [Event Name]: [catchy one-liner]
The one-liner MUST be original and specific to this fight. Reference a concrete detail: a stat, a streak, a style clash, a fighter's signature move, or a narrative unique to this matchup. NEVER use generic phrases like "Youth vs. Experience", "Clash of Styles", "Battle of [X]", "Proving Ground", "Unfinished Business", or any cliché. Think like a creative sportswriter who would be embarrassed by a generic headline.

## Overview
2-3 sentences setting up the matchup.

## Tale of the Tape
A markdown table comparing key attributes: record, age, height, reach, stance.

## [Red Fighter Name]
Analysis of recent form using their last few fights. Reference their skill profile from the rating data, backed by concrete stats from recent fights. You can include a table of recent results if it supports your point.

## [Blue Fighter Name]
Same format as above.

## Key Factors
Write this as flowing prose, not a bulleted list. Weave 3-5 factors together into a cohesive paragraph or two that tells the story of how this fight will be decided. Use the matchup edges data to identify the biggest skill differentials, and ground each one in concrete stats: striking rates, takedown numbers, finish rates, knockdowns, submission attempts, control time, age, reach advantages, quality of opposition. The factors should build a narrative that supports the predicted winner. You can acknowledge the opponent's strengths but frame them as insufficient to overcome the pick.

## How This Fight Plays Out
A short narrative (3-4 sentences) describing how you see the fight unfolding. Your analysis MUST build the case for the predicted winner and predicted method throughout. Do not hedge, present the other fighter as equally likely, or undermine the prediction. You are making a confident pick. Include how it compares to the betting odds and note any value gaps between the model and the market.

**Prediction: [Fighter Last Name] by [Method].**
This line must appear as its own paragraph at the very end, fully bolded. It is not a section header. The fighter and method MUST match the MODEL PREDICTION and METHOD PREDICTION provided in the data.

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
{json.dumps(context['odds'], indent=2) if context['odds'] else 'No odds available'}

{_glicko_block('RED', context.get('red_glicko'))}

{_glicko_block('BLUE', context.get('blue_glicko'))}

{_glicko_matchup_edges(context.get('red_glicko'), context.get('blue_glicko'))}"""

    return system, user


def generate_preview(fight_id: int, db: Session, force: bool = False) -> UFCFightPreview | None:
    """Generate an AI preview for a single fight."""
    if not settings.DEEPSEEK_API_KEY:
        log.warning("DEEPSEEK_API_KEY not set, skipping preview generation")
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
        client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
        response = client.chat.completions.create(
            model=settings.PREVIEW_MODEL,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        content = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        if existing:
            existing.content = content
            existing.model_used = settings.PREVIEW_MODEL
            existing.prompt_tokens = prompt_tokens
            existing.completion_tokens = completion_tokens
        else:
            existing = UFCFightPreview(
                fight_id=fight_id,
                content=content,
                model_used=settings.PREVIEW_MODEL,
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
