"""The fact sheet: every number an article may use, all from our database.

The writer gets the same stats block the single-call previews use
(preview_service.build_preview_prompt), plus the market consensus and the two fighters'
earlier meetings. The checker gets `allowed`: every number in that sheet together with
the forms a writer naturally turns them into (0.645 -> 64.5% / 65%, 420 days -> 14 months,
45/120 -> 38%).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

METHOD_LABEL = {"KO/TKO": "KO/TKO", "KO": "KO/TKO", "Submission": "Submission", "SUB": "Submission",
                "Decision": "Decision", "DEC": "Decision"}


@dataclass
class FactSheet:
    fight_id: int
    context: dict
    stats_prompt: str  # the user message the single-call previews send, plus extras
    system_prompt: str
    red: dict
    blue: dict
    event_name: str
    card_position: int | None
    is_title: bool
    pick_corner: str | None
    pick_name: str | None
    pick_last: str | None
    pick_prob: float | None
    method: str | None
    market_prob: float | None  # the pick's no-vig market probability
    meetings: list[dict] = field(default_factory=list)
    allowed: set[float] = field(default_factory=set)

    @property
    def edge_pts(self) -> float | None:
        if self.pick_prob is None or self.market_prob is None:
            return None
        return round((self.pick_prob - self.market_prob) * 100, 1)

    @property
    def pick_key(self) -> str:
        return f"{self.pick_corner}:{self.method}"


NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    return [float(n) for n in NUM.findall(text or "")]


def _expand(values: list[float]) -> set[float]:
    """The forms a writer turns a number into."""
    out: set[float] = set()
    for v in values:
        v = abs(v)
        out |= {v, round(v), round(v, 1)}
        if 0 < v < 1:  # probabilities and rates
            out |= {round(v * 100), round(v * 100, 1), round((1 - v) * 100), round((1 - v) * 100, 1)}
        if v >= 7:  # days -> weeks / months / years
            out |= {round(v / 7), round(v / 30), int(v // 30), round(v / 365, 1), round(v / 365), round(v / 60)}
    return out


def _ratios(text: str) -> set[float]:
    """45/120 -> 37.5, 38 (accuracy written as a percentage)."""
    out = set()
    for a, b in re.findall(r"(\d+)/(\d+)", text):
        a, b = int(a), int(b)
        if b:
            p = a / b * 100
            out |= {round(p), round(p, 1)}
        out.add(b - a)
    return out


def _sums(context: dict) -> set[float]:
    """Totals over the last few fights ("14 takedowns across his last five")."""
    out = set()
    for key in ("red_recent_fights", "blue_recent_fights"):
        fights = [f.get("stats") or {} for f in context.get(key) or []]
        for n in range(1, len(fights) + 1):
            window = fights[:n]
            for stat in ("kd", "sub_att"):
                out.add(sum(int(f.get(stat) or 0) for f in window))
            for stat in ("sig_str", "td"):
                landed = [int(f[stat].split("/")[0]) for f in window if f.get(stat)]
                att = [int(f[stat].split("/")[1]) for f in window if f.get(stat)]
                if landed:
                    out |= {sum(landed), sum(att), round(sum(landed) / len(landed)), round(sum(landed) / len(landed), 1)}
                    if sum(att):
                        p = sum(landed) / sum(att) * 100
                        out |= {round(p), round(p, 1)}
    return out


def _meetings(db: Session, fight, red_id: int, blue_id: int) -> list[dict]:
    from app.models.ufc import UFCEvent, UFCFight
    rows = (db.query(UFCFight, UFCEvent).join(UFCEvent, UFCEvent.id == UFCFight.event_id)
            .filter(UFCFight.id != fight.id, UFCFight.winner_id.isnot(None) | UFCFight.method.isnot(None),
                    or_(and_(UFCFight.red_fighter_id == red_id, UFCFight.blue_fighter_id == blue_id),
                        and_(UFCFight.red_fighter_id == blue_id, UFCFight.blue_fighter_id == red_id)))
            .order_by(UFCEvent.date).all())
    out = []
    for f, e in rows:
        winner = None
        if f.winner_id:
            w = f.red_fighter if f.winner_id == f.red_fighter_id else f.blue_fighter
            winner = f"{w.first_name} {w.last_name}"
        out.append({"event": e.name, "date": str(e.date), "winner": winner, "method": f.method,
                    "round": f.finish_round, "time": f.finish_time, "details": f.details})
    return out


def _last_fight_times(db: Session, fighter_id: int, limit: int = 5) -> list[dict]:
    from app.models.ufc import UFCFight
    rows = (db.query(UFCFight).filter(or_(UFCFight.red_fighter_id == fighter_id,
                                          UFCFight.blue_fighter_id == fighter_id),
                                      UFCFight.method.isnot(None))
            .order_by(UFCFight.date.desc()).limit(limit).all())
    return [{"date": str(f.date), "round": f.finish_round, "time": f.finish_time, "details": f.details}
            for f in rows]


def _market_red_prob(db: Session, fight_id: int, context: dict) -> float | None:
    try:
        from app.services.ufc.prediction_markets.serving import consensus_prob
        c = consensus_prob(db, fight_id)
        if c and c.get("red_prob"):
            return float(c["red_prob"])
    except Exception as e:
        log.info(f"No exchange consensus for {fight_id}: {e}")
    # Fall back to the sportsbooks: mean no-vig probability across books.
    probs = []
    for o in context.get("odds") or []:
        try:
            r, b = int(o["red_odds"]), int(o["blue_odds"])
        except (TypeError, ValueError):
            continue
        imp = lambda a: 100 / (a + 100) if a > 0 else -a / (-a + 100)
        pr, pb = imp(r), imp(b)
        probs.append(pr / (pr + pb))
    return sum(probs) / len(probs) if probs else None


def build(db: Session, fight_id: int) -> FactSheet | None:
    from app.models.ufc import UFCFight
    from app.services.ufc.preview_service import build_preview_prompt, gather_fight_context

    fight = db.get(UFCFight, fight_id)
    context = gather_fight_context(fight_id, db)
    if fight is None or context is None or not context.get("prediction"):
        return None
    system, user = build_preview_prompt(context)
    red, blue = fight.red_fighter, fight.blue_fighter
    rd = {"id": red.id, "name": f"{red.first_name} {red.last_name}", "last": red.last_name}
    bd = {"id": blue.id, "name": f"{blue.first_name} {blue.last_name}", "last": blue.last_name}

    pred, mp = context["prediction"], context.get("method_prediction")
    corner = pred["predicted_winner"]
    pick = rd if corner == "red" else bd
    pick_prob = pred["red_prob"] if corner == "red" else 1 - pred["red_prob"]
    method = METHOD_LABEL.get(mp["predicted_method"], mp["predicted_method"]) if mp else None
    red_mkt = _market_red_prob(db, fight_id, context)
    market = None if red_mkt is None else (red_mkt if corner == "red" else 1 - red_mkt)

    meetings = _meetings(db, fight, red.id, blue.id)
    extra = [f"\n**MARKET (no-vig consensus):** {pick['name']} {market * 100:.1f}%"
             f" (model {pick_prob * 100:.1f}%, edge {(pick_prob - market) * 100:+.1f} pts)"
             if market is not None else "\n**MARKET:** No market price yet"]
    if meetings:
        extra.append("\n**PREVIOUS MEETINGS (these two fighters):**")
        for m in meetings:
            extra.append(f"- {m['event']} ({m['date']}): {m['winner'] or 'No winner'} by {m['method']}, "
                         f"R{m['round']} {m['time']}" + (f" ({m['details']})" if m['details'] else ""))
    for d, key in ((rd, "red"), (bd, "blue")):
        times = _last_fight_times(db, d["id"])
        if times:
            extra.append(f"\n**{d['name'].upper()} RECENT FINISH TIMES:** " + "; ".join(
                f"{t['date']}: R{t['round']} {t['time']}" for t in times))
    stats_prompt = user + "\n" + "\n".join(extra)

    nums = numbers_in(stats_prompt)
    allowed = _expand(nums) | _ratios(stats_prompt) | _sums(context)
    # Times like 0:26 -> 26 seconds, 4:12 -> 252 seconds.
    for m, s in re.findall(r"\b(\d+):(\d{2})\b", stats_prompt):
        allowed |= {int(m) * 60 + int(s), int(s), int(m)}
    if market is not None:
        allowed |= _expand([market, pick_prob - market, abs(pick_prob - market) * 100])
        allowed |= {round(abs(pick_prob - market) * 100, 1), round(abs(pick_prob - market) * 100)}

    return FactSheet(
        fight_id=fight_id, context=context, stats_prompt=stats_prompt, system_prompt=system,
        red=rd, blue=bd, event_name=context["event"]["name"], card_position=fight.card_position,
        is_title="title" in (fight.weight_class or "").lower(), pick_corner=corner,
        pick_name=pick["name"], pick_last=pick["last"], pick_prob=pick_prob, method=method,
        market_prob=market, meetings=meetings, allowed=allowed,
    )
