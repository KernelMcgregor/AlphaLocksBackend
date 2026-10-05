"""Shared plumbing for the cross-division rankings (BMF and P4P).

Neither ranking is Tapology's, and neither is fitted to anyone's published output. Both
are opinionated composites: each fighter gets a handful of 0-100 components, and the
page recombines them with the viewer's own weights. What lives here is the part both
share — who is eligible, how a bout's age and opposition discount it, the opponent-
quality and activity components, and the one read of the database they both need.

Cross-division comparability is the whole problem. Two choices carry it:

  * Opponent quality is Tapology's opponent TIER, not a rating percentile. The
    resume -> tier line is absolute (see `tapology_rankings.TIER_INTERCEPT`), so a tier-8
    flyweight and a tier-8 heavyweight mean the same thing: an opponent who had been
    beating opponents who had themselves been winning.
  * Every other per-division quantity (KO rates, chin rates, Elo spread) is measured
    against the fighter's OWN division's baseline before it is put on a 0-100 scale,
    so heavyweights do not top the BMF list for being heavy.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

from app.services.ufc.fighter_registry import Eligibility, is_rankable

#: UFC bouts considered per fighter, newest first.
WINDOW = 10
#: A bout's weight halves every this many years.
HALF_LIFE_YEARS = 3.0
#: Quality weight of a win over a tier-1 opponent; a tier-10 is worth 1.0.
Q_FLOOR = 0.4
#: Tier assumed when a bout has none (the opponent's first UFC appearance predates data).
DEFAULT_TIER = 5
#: Opponent tier at which a win counts as a "ranked-calibre" win for the resume bonus.
ELITE_TIER = 8
ELITE_WIN_BONUS, ELITE_WIN_CAP = 0.5, 1.5
#: Result credit toward opponent quality: facing elite opposition counts even in defeat.
RESULT_CREDIT = {"W": 1.0, "D": 0.6, "L": 0.4}

#: 4 results, not Tapology's 1: "proven" is the point of both lists. 699 days is the
#: registry's standard inactivity bound (21 months plus Tapology's 60-day grace).
ALT_ELIGIBILITY = Eligibility(
    min_decided_fights=4, min_rounds=0, max_days_inactive=699, count_all_results=True,
)

POOLS = ("men", "women")


def pool_of(division: str) -> str:
    return "women" if division.startswith("w_") else "men"


def recency_weight(days_ago: float) -> float:
    return 0.5 ** (max(days_ago, 0.0) / 365.25 / HALF_LIFE_YEARS)


def quality_weight(tier: int) -> float:
    t = (max(1, min(10, tier)) - 1) / 9
    return Q_FLOOR + (1.0 - Q_FLOOR) * t


def recency_score(days_since_last: int, bouts_last_730d: int) -> float:
    """Activity, 0-100 on an ABSOLUTE scale — not a percentile, so it means the same
    thing next year: fought last month and three times in two years ~ 100."""
    return 100.0 * (0.7 * 0.5 ** (max(days_since_last, 0) / 365.0)
                    + 0.3 * min(bouts_last_730d, 3) / 3)


#: Opposition raw value that maps to 100: an average opponent tier of 9 at full win credit.
Q_REF = 9.0


def scale_quality(raw: float) -> float:
    """Opposition, 0-100 on an ABSOLUTE scale.

    Not a percentile. Percentiles were tried first and crushed the top end: the median
    eligible fighter's opposition averages tier ~1.8 and the 95th percentile ~5.5, so
    every contender who matters landed between 95 and 100 and the component stopped
    separating them. Linear in mean tier keeps Makhachev's 8.9 distinct from a 5.5.
    """
    return round(100.0 * max(0.0, min(raw / Q_REF, 1.0)), 2)


@dataclass
class BoutView:
    """One UFC bout from one fighter's side, with everything both rankings read."""

    fight_id: int
    date: date
    opponent_id: int
    result: str                  # "W" | "L" | "D"
    outcome: str                 # outcome_types: ko/sub/ud/md/sd/doctor/injury/draw
    finish_round: int | None
    division: str
    tier: int
    kd: int = 0                  # knockdowns landed
    kd_abs: int = 0              # knockdowns absorbed
    head_abs: int = 0            # significant head strikes absorbed
    w: float = 1.0               # recency weight
    q: float = 1.0               # quality weight

    @property
    def went_distance(self) -> bool:
        return self.outcome in ("ud", "md", "sd", "draw")


@dataclass
class FighterView:
    fighter_id: int
    name: str
    division: str
    pool: str
    bouts: list[BoutView]        # newest first, at most WINDOW
    last_activity: date
    bouts_last_730d: int
    is_champion: bool = False
    extra: dict = field(default_factory=dict)


def opp_quality_raw(bouts: list[BoutView]) -> float:
    """Recency-weighted mean opponent tier, credited by result, plus a capped bonus for
    wins over elite-tier opposition."""
    tw = sum(b.w for b in bouts)
    if tw <= 0:
        return 0.0
    base = sum(b.w * b.tier * RESULT_CREDIT[b.result] for b in bouts) / tw
    elite = sum(1 for b in bouts if b.result == "W" and b.tier >= ELITE_TIER)
    return base + min(ELITE_WIN_CAP, ELITE_WIN_BONUS * elite)


def ledger_entry(b: BoutView, names: dict[int, str], **extra) -> dict:
    return {
        "fight_id": str(b.fight_id), "date": b.date.isoformat(),
        "opponent_id": str(b.opponent_id), "opponent_name": names.get(b.opponent_id, ""),
        "result": b.result, "outcome": b.outcome, "round": b.finish_round,
        "division": b.division, "tier": b.tier, "kd": b.kd, "kd_abs": b.kd_abs,
        "w": round(b.w, 3), "q": round(b.q, 3), **extra,
    }


# ---------------------------------------------------------------------------
# Database read — the one pass both rankings share
# ---------------------------------------------------------------------------

def load_context(db, as_of: date | None = None) -> dict:
    """Read everything once. Read-only."""
    from app.models.ufc import UFCFight, UFCFightStats
    from app.services.ufc.fighter_registry import build_fighter_registry
    from app.services.ufc.outcome_types import classify_outcome
    from app.services.ufc.tapology_rankings import build_history

    today = as_of or date.today()
    hist = build_history(db)
    registry = build_fighter_registry(db, as_of=today)

    fights = {
        r[0]: r for r in db.query(
            UFCFight.id, UFCFight.date, UFCFight.red_fighter_id, UFCFight.blue_fighter_id,
            UFCFight.winner_id, UFCFight.method, UFCFight.details,
            UFCFight.fight_time_seconds,
        ).all()
    }
    totals: dict[tuple[int, int], dict] = {}
    for fight_id, fighter_id, kd, head, sig, td, ctrl, sub, sig_att, td_att in db.query(
        UFCFightStats.fight_id, UFCFightStats.fighter_id, UFCFightStats.kd,
        UFCFightStats.head_landed, UFCFightStats.sig_str_landed, UFCFightStats.td_landed,
        UFCFightStats.ctrl_seconds, UFCFightStats.sub_att, UFCFightStats.sig_str_attempted,
        UFCFightStats.td_attempted,
    ).filter(UFCFightStats.round_number == 0).all():
        totals[(fight_id, fighter_id)] = {
            "kd": kd or 0, "head_landed": head or 0, "sig_str_landed": sig or 0,
            "td_landed": td or 0, "ctrl_seconds": ctrl or 0, "sub_att": sub or 0,
            "sig_str_attempted": sig_att or 0, "td_attempted": td_att or 0,
        }

    #: Fight ids of UFC title bouts, undisputed or interim (not tournament finals).
    title_fights = {
        fid for fid, wc in db.query(UFCFight.id, UFCFight.weight_class).all()
        if wc and wc.startswith("UFC ") and "title bout" in wc.lower()
        and "tournament" not in wc.lower()
    }

    outcome = {}
    for fid, row in fights.items():
        outcome[fid] = classify_outcome(row[5], row[6], row[4])

    champs = champions_at(hist, today)
    from app.services.ufc.p4p_rankings import division_context
    divctx = division_context(hist, today)

    return {
        "today": today, "hist": hist, "registry": registry, "fights": fights,
        "totals": totals, "outcome": outcome, "champions": champs,
        "title_fights": title_fights, **divctx,
    }


def champions_at(hist: dict, as_of: date) -> dict[str, int]:
    """Belt holders as of `as_of`, judged the way the Tapology ranker judges them.

    A champion counts as having left a division only when neither of their last two
    bouts was in it. The registry's single division is the wrong input here: it takes two
    bouts to move, so a fighter who wins a belt in their FIRST bout at a new weight —
    Topuria at lightweight, Makhachev at welterweight — would be ruled out of the
    division they just won.
    """
    from app.services.ufc.champions import champions_as_of
    from app.services.ufc.tapology_rankings import _divisions_as_of

    return champions_as_of(hist["title_bouts"], as_of,
                           _divisions_as_of(hist["div_bouts"], as_of))


def bout_views(ctx: dict, fid: int, before: date, n: int | None = WINDOW) -> list[BoutView]:
    """A fighter's last `n` scored UFC bouts strictly before `before`, newest first.
    `n=None` returns all of them."""
    own = ctx["hist"]["bouts"].get(fid, [])
    lo = bisect_left([b.date for b in own], before)
    chosen = own[:lo] if n is None else own[max(0, lo - n):lo]
    out = []
    today = ctx["today"]
    for b in reversed(chosen):
        tier = ctx["hist"]["tiers"].get((fid, b.fight_id), DEFAULT_TIER)
        me = ctx["totals"].get((b.fight_id, fid), {})
        opp = ctx["totals"].get((b.fight_id, b.opponent_id), {})
        out.append(BoutView(
            fight_id=b.fight_id, date=b.date, opponent_id=b.opponent_id,
            result="W" if b.won else ("D" if b.drew else "L"),
            outcome=ctx["outcome"].get(b.fight_id, "void"),
            finish_round=b.finish_round, division=b.division, tier=tier,
            kd=me.get("kd", 0), kd_abs=opp.get("kd", 0),
            head_abs=opp.get("head_landed", 0),
            w=recency_weight((today - b.date).days), q=quality_weight(tier),
        ))
    return out


def is_eligible(st, today: date, is_champion: bool) -> bool:
    """The one eligibility predicate, shared by the rankers and the publisher's check.

    A reigning champion is always eligible: the 4-fight floor exists to keep unproven
    prospects off the list, and a belt is proof. Kayla Harrison won hers in her third UFC
    bout."""
    if is_champion:
        return st.division != "unknown" and (today - st.last_activity).days \
            <= ALT_ELIGIBILITY.max_days_inactive
    return is_rankable(st, today, ALT_ELIGIBILITY)


def eligible_fighters(ctx: dict) -> list[FighterView]:
    """Everyone both lists may rank, with their bout window built.

    A champion is filed under the division of their belt; everyone else under the
    registry's division (which needs two bouts at a new weight to move them)."""
    from app.services.ufc.tapology_rankings import EXCLUDE_STATUSES

    today = ctx["today"]
    cutoff = today + timedelta(days=1)
    status = ctx["hist"]["status"]
    names = ctx["hist"]["names"]
    champ_div = {fid: d for d, fid in ctx["champions"].items()}
    # `status` is present-tense (who is retired NOW), so it only applies to a current
    # ranking — the same rule tapology_rankings uses for historical replays.
    apply_status = (date.today() - today).days <= 60
    out = []
    for fid, st in ctx["registry"].items():
        if apply_status and status.get(fid) in EXCLUDE_STATUSES:
            continue
        if not is_eligible(st, today, fid in champ_div):
            continue
        bouts = bout_views(ctx, fid, cutoff)
        if not bouts:
            continue
        acts = ctx["hist"]["activity"].get(fid, [])
        recent = sum(1 for d in acts if 0 <= (today - d).days <= 730)
        division = champ_div.get(fid, st.division)
        out.append(FighterView(
            fighter_id=fid, name=names.get(fid, ""), division=division,
            pool=pool_of(division), bouts=bouts, last_activity=st.last_activity,
            bouts_last_730d=recent, is_champion=fid in champ_div,
        ))
    return out


def safe_log(x: float) -> float:
    return math.log(max(x, 1e-9))


def group_by(items, key) -> dict:
    out = defaultdict(list)
    for it in items:
        out[key(it)].append(it)
    return out
