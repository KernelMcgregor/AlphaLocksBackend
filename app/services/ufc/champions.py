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
    as_of = as_of or date.today()
    rows = (
        db.query(UFCFight)
        .join(UFCEvent, UFCFight.event_id == UFCEvent.id)
        .order_by(UFCFight.date.desc(), UFCFight.id.desc())
        .all()
    )

    # Every title-bout win, undisputed or interim, oldest first. Used to identify the
    # sitting champion when a title defence produces no result.
    title_wins: dict[str, list[tuple[date, int]]] = {}
    for f in reversed(rows):
        if not f.date or f.date > as_of or not f.weight_class:
            continue
        low = f.weight_class.lower()
        if not f.weight_class.startswith("UFC ") or "title bout" not in low:
            continue
        if "tournament" in low:
            continue
        if is_decided(f.method, f.winner_id):
            div = classify_weight_class(f.weight_class)
            if div != "unknown":
                title_wins.setdefault(div, []).append((f.date, f.winner_id))

    champs: dict[str, int] = {}
    for f in rows:
        if not f.date or f.date > as_of:
            continue
        if not is_undisputed_title_bout(f.weight_class):
            continue
        div = classify_weight_class(f.weight_class)
        if div == "unknown" or div in champs:
            continue

        if is_decided(f.method, f.winner_id):
            champs[div] = f.winner_id
            continue

        # A waved-off title fight does not transfer the belt, and it is also the only
        # trace in this data of an *elevation*: Tom Aspinall was promoted from interim to
        # undisputed with no bout to mark it, so the last decided undisputed heavyweight
        # title bout is still Jones-Miocic and a naive reading crowns Jon Jones.
        #
        # The champion is whichever participant had most recently won a title bout in the
        # division — interim counts, since that is exactly what an elevation promotes.
        prior = [w for w in title_wins.get(div, []) if w[0] < f.date]
        for when, winner_id in reversed(prior):
            if winner_id in (f.red_fighter_id, f.blue_fighter_id):
                champs[div] = winner_id
                log.info(f"  Champions: {div} title bout on {f.date} had no result; "
                         f"belt stays with the prior title winner ({when})")
                break

    if divisions is not None:
        vacated = {d: fid for d, fid in champs.items()
                   if divisions.get(fid) not in (None, d)}
        for d in vacated:
            champs.pop(d, None)
        if vacated:
            log.info(f"  Champions: {len(vacated)} belt(s) treated as vacant "
                     f"(holder moved divisions): {sorted(vacated)}")

    return champs
