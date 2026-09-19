"""Who currently holds each belt, derived from title-bout results.

There is no `is_title` column and no champion table. The only signal is the bout name in
`ufc_fights.weight_class`, which is why the old ranker sniffed for the substring "title" —
a check that also matched interim bouts and Road to UFC tournament finals.

This module does the derivation properly, and it is a derivation rather than a curated
list on purpose: a hand-maintained champion table goes stale the moment a belt changes
hands on a card nobody remembered to update.

Tapology's rule, which this implements: **the most recent undisputed (non-interim)
champion is locked at #1**, with no separate "C" designation.
"""

from __future__ import annotations

import logging
from datetime import date

from app.models.ufc import UFCEvent, UFCFight
from app.services.ufc.fighter_registry import classify_weight_class, is_decided

log = logging.getLogger("champions")


def is_undisputed_title_bout(weight_class: str | None) -> bool:
    """A real UFC championship bout, not an interim or a tournament final.

    Real ones are named `UFC <Division> Title Bout`. Requiring the `UFC ` prefix is what
    excludes `Road to UFC 4 Flyweight Tournament Title Bout` and
    `Ultimate Fighter 33 Welterweight Tournament Title Bout`, both of which contain
    "Title Bout" and neither of which awards a UFC belt.
    """
    if not weight_class:
        return False
    wc = weight_class.strip()
    if not wc.startswith("UFC "):
        return False
    low = wc.lower()
    if "title bout" not in low:
        return False
    # Tapology counts only undisputed belts; an interim champion is not pinned.
    return "interim" not in low and "tournament" not in low


def load_title_bouts(db) -> list[tuple]:
    """Every UFC title bout, newest first, as plain tuples.

    Split out so a caller that needs champions at MANY dates — the rank-history backfill
    walks ~700 event dates — pays for this query once instead of once per date. Columns
    only: hydrating the full fight table as ORM objects was the dominant cost.
    """
    rows = (
        db.query(
            UFCFight.date, UFCFight.weight_class, UFCFight.method,
            UFCFight.winner_id, UFCFight.red_fighter_id, UFCFight.blue_fighter_id,
        )
        .join(UFCEvent, UFCFight.event_id == UFCEvent.id)
        .order_by(UFCFight.date.desc(), UFCFight.id.desc())
        .all()
    )
    return [r for r in rows if r[0] and r[1]]


def current_champions(db, as_of: date | None = None,
                      divisions: dict[int, str] | None = None) -> dict[str, int]:
    """division -> fighter_id of the reigning undisputed champion.

    Two rules that matter, both of which a naive "latest title bout winner" gets wrong:

    * **A no-contest does not change the belt.** The most recent undisputed heavyweight
      title bout (2025-10-25) was waved off with no winner; the champion is still whoever
      won the one before it. So this walks backwards to the most recent *decided* title
      bout in each division.
    * **A champion who has moved divisions is not still champion of the old one.** Islam
      Makhachev won the lightweight belt and then moved to welterweight; the lightweight
      title is vacant, not his. If `divisions` is supplied, a champion-of-record who no
      longer competes in that division is dropped rather than pinned.
    """
    return champions_as_of(load_title_bouts(db), as_of or date.today(), divisions)


def champions_as_of(title_bouts: list[tuple], as_of: date,
                    divisions: dict | None = None) -> dict[str, int]:
    """`current_champions` as a pure function over prefetched rows.

    Same rules, no database. Takes the output of `load_title_bouts` so the backfill can
    resolve the champion at every historical date from one query.
    """
    rows = title_bouts

    # Every title-bout win, undisputed or interim, oldest first. Used to identify the
    # sitting champion when a title defence produces no result.
    title_wins: dict[str, list[tuple[date, int]]] = {}
    for f_date, wc, method, winner_id, _red, _blue in reversed(rows):
        if f_date > as_of:
            continue
        low = wc.lower()
        if not wc.startswith("UFC ") or "title bout" not in low:
            continue
        if "tournament" in low:
            continue
        if is_decided(method, winner_id):
            div = classify_weight_class(wc)
            if div != "unknown":
                title_wins.setdefault(div, []).append((f_date, winner_id))

    champs: dict[str, int] = {}
    for f_date, wc, method, winner_id, red_id, blue_id in rows:
        if f_date > as_of:
            continue
        if not is_undisputed_title_bout(wc):
            continue
        div = classify_weight_class(wc)
        if div == "unknown" or div in champs:
            continue

        if is_decided(method, winner_id):
            champs[div] = winner_id
            continue

        # A waved-off title fight does not transfer the belt, and it is also the only
        # trace in this data of an *elevation*: Tom Aspinall was promoted from interim to
        # undisputed with no bout to mark it, so the last decided undisputed heavyweight
        # title bout is still Jones-Miocic and a naive reading crowns Jon Jones.
        #
        # The champion is whichever participant had most recently won a title bout in the
        # division — interim counts, since that is exactly what an elevation promotes.
        prior = [w for w in title_wins.get(div, []) if w[0] < f_date]
        for when, prior_winner in reversed(prior):
            if prior_winner in (red_id, blue_id):
                champs[div] = prior_winner
                log.debug(f"  Champions: {div} title bout on {f_date} had no result; "
                          f"belt stays with the prior title winner ({when})")
                break

    if divisions is not None:
        # `divisions` maps a fighter to the class(es) they currently compete in. Tapology
        # ranks a fighter in BOTH when their last two bouts differ, so the value may be a
        # collection — and comparing a list against a division string with `not in
        # (None, d)` silently declared every belt vacant.
        def _still_competes(fid: int, d: str) -> bool:
            current = divisions.get(fid)
            if current is None:
                return True          # unknown division is not evidence of a move
            if isinstance(current, (list, set, tuple)):
                return d in current
            return current == d

        vacated = {d: fid for d, fid in champs.items() if not _still_competes(fid, d)}
        for d in vacated:
            champs.pop(d, None)
        if vacated:
            log.info(f"  Champions: {len(vacated)} belt(s) treated as vacant "
                     f"(holder moved divisions): {sorted(vacated)}")

    return champs
