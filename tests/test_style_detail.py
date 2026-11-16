"""Per-fighter style detail (routers/ufc._build_style_detail): families and lineage."""
from __future__ import annotations

import json

import pytest

from app.config import settings

pytestmark = pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"),
                                reason="needs a scratch SQLite DATABASE_URL")


def _style(fid, family, prime, ancestors=(), notable=()):
    from app.models.ufc import UFCFighterStyle
    return UFCFighterStyle(
        fighter_id=fid, family=family, division="lightweight", x=0.5, y=0.5,
        first_year=int(prime) - 3, last_year=int(prime) + 3, prime_year=prime, active=False,
        quality=60.0, notable=bool(notable), profile=json.dumps({"slpm": 70}),
        ancestors=json.dumps([{"id": a, "sim": s} for a, s in ancestors]),
        notable_ancestors=json.dumps([{"id": a, "sim": s} for a, s in notable]),
        map_version="v5")


@pytest.fixture()
def db():
    from app.database import Base, SessionLocal, engine
    from app.models.ufc import UFCFighter, UFCFighterSimilarity, UFCFighterStyle, UFCStyleFamily

    tables = [t.__table__ for t in (UFCFighter, UFCStyleFamily, UFCFighterStyle, UFCFighterSimilarity)]
    Base.metadata.drop_all(engine, tables=list(reversed(tables)))
    Base.metadata.create_all(engine, tables=tables)
    s = SessionLocal()
    s.add_all([UFCFighter(id=i, ufcstats_id=str(i), first_name=f, last_name=l)
               for i, f, l in [(1, "Old", "Legend"), (2, "Middle", "Gen"), (3, "Young", "Gun"), (4, "New", "Kid")]])
    s.add_all([UFCStyleFamily(id=i, name=n, blurb="b", traits="[]", profile="{}", n_fighters=1, map_version="v5")
               for i, n in [(0, "Range strikers"), (1, "Wrestle-control")]])
    s.flush()
    # 1 <- 2 <- 3 <- 4: each descends from the one before.
    s.add_all([_style(1, 0, 2004.0), _style(2, 0, 2012.0, [(1, 0.8)], [(1, 0.8)]),
               _style(3, 1, 2019.0, [(2, 0.7)]), _style(4, 1, 2025.0, [(3, 0.9), (2, 0.5)])])
    s.commit()
    yield s
    s.close()


def test_detail_carries_families_and_two_generations(db):
    from app.routers.ufc import _build_style_detail
    d = _build_style_detail(db, 3)
    assert [f["name"] for f in d["families"]] == ["Range strikers", "Wrestle-control"]
    assert "profile" not in d["families"][0]
    names = {n["id"]: n["name"] for n in d["lineage"]["nodes"]}
    # Parent and grandparent resolved, plus the heir.
    assert names == {"2": "Middle Gen", "1": "Old Legend", "4": "New Kid"}
    assert d["lineage"]["heirs"] == [{"id": "4", "sim": 0.9}]
    assert d["lineage"]["notable_heirs"] == []


def test_heirs_sorted_and_notable(db):
    from app.routers.ufc import _lineage
    l2 = _lineage(db, 2)
    assert [h["id"] for h in l2["heirs"]] == ["3", "4"]
    assert l2["notable_heirs"] == []
    assert _lineage(db, 1)["notable_heirs"] == [{"id": "2", "sim": 0.8}]
    assert _lineage(db, 99) == {"nodes": [], "heirs": [], "notable_heirs": []}
