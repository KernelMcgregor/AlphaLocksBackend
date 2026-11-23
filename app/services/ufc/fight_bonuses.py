"""Fight of the Night / Performance of the Night (and the older KO / Submission of the
Night) from the ufcstats bonus icons.

The icons sit beside the bout on both the event page (in the bout row) and the fight page
(in the fight title, next to the title belt), so the nightly scrape reads them off the
fight page and the backfill reads a whole card off its event page.
"""
from __future__ import annotations

import logging
import re

from sqlalchemy.orm import Session

from app.models.ufc import UFCFight
from app.models.ufc_highlights import UFCFightBonus

log = logging.getLogger(__name__)

ICONS = {"fight.png": "fotn", "perf.png": "potn", "ko.png": "kotn", "sub.png": "sotn"}
BONUS_TYPES = tuple(ICONS.values())
_FIGHT_ID = re.compile(r"fight-details/([0-9a-f]{16})")


def icons_in(el) -> list[str]:
    """Bonus types among the <img> icons under `el`, in page order (the belt is ignored)."""
    out = []
    for img in el.select("img"):
        kind = ICONS.get(img.get("src", "").rsplit("/", 1)[-1])
        if kind and kind not in out:
            out.append(kind)
    return out


def event_page_bonuses(soup) -> dict[str, list[str]]:
    """{fight ufcstats_id: [bonus types]} for every bout row of an event page."""
    out = {}
    for tr in soup.select("tbody tr.b-fight-details__table-row"):
        m = _FIGHT_ID.search(tr.get("data-link", ""))
        if m:
            out[m.group(1)] = icons_in(tr)
    return out


def sync_fight_bonuses(db: Session, fight: UFCFight, kinds: list[str]) -> None:
    """Make the fight's bonus rows match `kinds` exactly (no commit)."""
    have = {b.bonus_type: b for b in db.query(UFCFightBonus).filter(UFCFightBonus.fight_id == fight.id)}
    for kind, row in have.items():
        if kind not in kinds:
            db.delete(row)
    for kind in kinds:
        who = None if kind == "fotn" else fight.winner_id
        row = have.get(kind)
        if row is None:
            db.add(UFCFightBonus(fight_id=fight.id, bonus_type=kind, fighter_id=who))
        elif row.fighter_id != who:
            row.fighter_id = who


def try_sync(db: Session, fight: UFCFight, kinds: list[str]) -> None:
    """sync_fight_bonuses in a savepoint, so a missing table or a bad row never costs the
    results scrape that called it."""
    try:
        with db.begin_nested():
            sync_fight_bonuses(db, fight, kinds)
    except Exception as e:  # noqa: BLE001 - bonuses are optional to the scrape
        log.warning(f"bonus sync skipped for fight {fight.id}: {e}")
