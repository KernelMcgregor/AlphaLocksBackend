"""Admin article editor (services/ufc/article_admin.py) on scratch SQLite."""
from __future__ import annotations

import base64
import datetime as dt
import io

import pytest

from app.config import settings

pytestmark = pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"),
                                reason="needs a scratch SQLite DATABASE_URL")


@pytest.fixture()
def db():
    from app.database import Base, SessionLocal, engine
    from app.models.ufc import (UFCArticleImage, UFCArticleMedia, UFCEvent, UFCFight, UFCFighter,
                                UFCFightPreview)

    tables = [t.__table__ for t in (UFCFighter, UFCEvent, UFCFight, UFCFightPreview, UFCArticleImage, UFCArticleMedia)]
    Base.metadata.drop_all(engine, tables=list(reversed(tables)))
    Base.metadata.create_all(engine, tables=tables)
    # SQLite doesn't autoincrement BigInteger keys (Postgres does): number new rows here.
    from sqlalchemy import event
    counter = iter(range(5000, 10**6))

    def _ids(mapper, conn, target):
        if target.id is None:
            target.id = next(counter)
    for model in (UFCArticleImage, UFCArticleMedia):
        if not event.contains(model, "before_insert", _ids):
            event.listen(model, "before_insert", _ids)
    s = SessionLocal()
    s.add_all([UFCFighter(id=1, ufcstats_id="a", first_name="Ciryl", last_name="Gane"),
               UFCFighter(id=2, ufcstats_id="b", first_name="Josh", last_name="Hokit"),
               UFCEvent(id=10, ufcstats_id="e", name="UFC 334: Gane vs. Hokit", date=dt.date.today())])
    s.flush()
    s.add(UFCFight(id=100, ufcstats_id="f", date=dt.date.today(), event_id=10, red_fighter_id=1, blue_fighter_id=2, card_position=0))
    s.flush()
    s.add(UFCFightPreview(id=1000, fight_id=100, content="# Gane vs. Hokit | UFC 334: Mirror\n\nBody.", model_used="x"))
    s.commit()
    yield s
    s.close()


def _png(w=900, h=600):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_save_marks_edited_and_keeps_notes(db):
    from app.services.ufc import article_admin as A
    A.save_article(db, 100, "# New\n\nText", "Bad blood since 2024.")
    a = A.get_article(db, 100)
    assert a["content"].startswith("# New") and a["editor_notes"] == "Bad blood since 2024." and a["edited_at"]
    with pytest.raises(ValueError):
        A.save_article(db, 100, "   ", None)


def test_cover_upload_and_frame(db):
    from app.services.ufc import article_admin as A
    A.upload_cover(db, 100, _png())
    c = A.get_article(db, 100)["cover"]
    assert c["phase"] == "manual" and c["width"] == 900
    A.frame_cover(db, 100, 0.1, 1.7)
    c = A.get_article(db, 100)["cover"]
    assert (c["focal_x"], c["focal_y"]) == (0.1, 1.0)
    A.remove_cover(db, 100)
    assert A.get_article(db, 100)["cover"] is None


def test_media_add_frame_delete_and_big_images_shrink(db):
    from app.services.ufc import article_admin as A
    mid = A.add_media(db, 100, _png(3000, 2000), "Faceoff")
    m = A.get_article(db, 100)["media"][0]
    assert m["id"] == mid and m["width"] == A.STORE_MAX_W and m["label"] == "Faceoff"
    A.frame_media(db, mid, 0.3, 0.2)
    assert A.get_article(db, 100)["media"][0]["focal_x"] == 0.3
    A.delete_media(db, mid)
    assert A.get_article(db, 100)["media"] == []


def test_rejects_non_image(db):
    from app.services.ufc import article_admin as A
    with pytest.raises(ValueError):
        A.add_media(db, 100, base64.b64encode(b"not an image").decode())
