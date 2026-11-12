"""What the public article pages read from the newsroom: the layout and earlier meetings.

Kept apart from the pipeline so the read-only routes import neither the OpenAI client nor
the scrapers. Every read tolerates the newsroom tables not existing yet.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


def layouts(db: Session, fight_ids: list[int]) -> dict[int, str]:
    """fight id -> layout, the editor's override first. Empty when nothing is stored."""
    if not fight_ids:
        return {}
    from app.models.ufc import UFCArticleRun
    try:
        rows = (db.query(UFCArticleRun.fight_id, UFCArticleRun.layout, UFCArticleRun.layout_override)
                .filter(UFCArticleRun.fight_id.in_(fight_ids)).all())
    except Exception as e:  # tables not created yet
        db.rollback()
        log.debug(f"No newsroom layouts: {e}")
        return {}
    return {fid: (override or layout) for fid, layout, override in rows if (override or layout)}


def meetings(db: Session, fight) -> list[dict]:
    """Earlier bouts between the two fighters, oldest first, for the chin-history block."""
    from app.services.ufc.newsroom.facts import _meetings
    return _meetings(db, fight, fight.red_fighter_id, fight.blue_fighter_id)
