"""Right-facing portraits (services/ufc/fighter_poses.py): parsing, matching, the endpoint."""
from __future__ import annotations

import datetime as dt
import io
from types import SimpleNamespace as NS

import pytest

from app.config import settings
from app.services.ufc import fighter_poses as P

# Trimmed from https://www.ufc.com/event/ufc-fight-night-november-21-2026 (2026-11-16).
_IMG = "https://ufc.com/images/styles/event_fight_card_upper_body_of_standing_athlete/s3/"


def _fight(red_slug, red_name_html, red_img, blue_slug, blue_name_html, blue_img):
    return f"""
<div class="c-listing-fight" data-fmid="1" data-status="">
  <div class="c-listing-fight__content"><div class="c-listing-fight__content-row">
    <div class="c-listing-fight__corner--red">
      <div class="c-listing-fight__corner-image--red">
        <a href="https://www.ufc.com/athlete/{red_slug}"><div class="layout"><div class="layout__region">
          <img src="{_IMG}{red_img}" width="185" height="540" alt="" loading="lazy" />
        </div></div></a>
      </div>
    </div>
    <div class="c-listing-fight__details">
      <div class="c-listing-fight__names-row">
        <div class="c-listing-fight__corner-name c-listing-fight__corner-name--red">
          <a href="https://www.ufc.com/athlete/{red_slug}">{red_name_html}</a>
        </div>
        <div class="c-listing-fight__vs">vs</div>
        <div class="c-listing-fight__corner-name c-listing-fight__corner-name--blue">
          <a href="https://www.ufc.com/athlete/{blue_slug}">{blue_name_html}</a>
        </div>
      </div>
    </div>
    <div class="c-listing-fight__corner--blue">
      <div class="c-listing-fight__corner-image--blue">
        <a href="https://www.ufc.com/athlete/{blue_slug}"><div class="layout"><div class="layout__region">
          <img src="{blue_img and _IMG + blue_img}" width="185" height="607" loading="lazy" />
        </div></div></a>
      </div>
    </div>
  </div></div>
</div>"""


EVENT_HTML = "<html><body><ul>" + "".join([
    _fight("jiri-prochazka", "Ji&#345;&#237; Proch&#225;zka", "2026-04/PROCHAZKA_JIRI_L_04-11.png?itok=vr41IWZw",
           "navajo-stirling",
           '<span class="c-listing-fight__corner-given-name">Navajo</span>\n'
           '<span class="c-listing-fight__corner-family-name">Stirling</span>',
           "2026-07/STIRLING_NAVAJO_R_08-01.png?itok=aL1Vfoli"),
    _fight("dan-hooker", "Dan Hooker", "2026-09/HOOKER_DAN_L_09-05.png?itok=q",
           "brian-ortega", "Brian Ortega", "2025-01/5/ORTEGA_BRIAN_R_09-14.png?itok=c"),
    # Red corner shot in the right-facing pose: still that fighter's right pose.
    _fight("kyoji-horiguchi", "Kyoji Horiguchi", "2025-11/HORIGUCHI_KYOJI_R_11-22.png?itok=q",
           "asu-almabayev", "Asu Almabayev", "2025-11/ALMABAYEV_ASU_R_11-22.png?itok=P"),
    # No `_R_` file on either side.
    _fight("jake-matthews", "Jake Matthews", "2026-05/MATTHEWS_JAKE_L_05-30.png?itok=8",
           "tahir-abdullayev", "Tahir Abdullayev", "2026-06/ABDULLAYEV_TAHIR_L_06-27.png?itok=p"),
    # The image is someone else's file: must never be stored against this fighter.
    _fight("aljamain-sterling", "Aljamain Sterling", "2026-04/STERLING_ALJAMAIN_L_04-25.png?itok=_",
           "kevin-vallejos", "Kevin Vallejos", "2025-12/SOMEONE_ELSE_R_12-13.png?itok=t"),
]) + "</ul></body></html>"


def _f(i, first, last, right=None):
    return NS(id=i, first_name=first, last_name=last, image_right_url=right)


ROSTER = [_f(1, "Jiri", "Prochazka"), _f(2, "Navajo", "Stirling"), _f(3, "Dan", "Hooker"),
          _f(4, "Brian", "Ortega"), _f(5, "Kyoji", "Horiguchi"), _f(6, "Asu", "Almabayev"),
          _f(7, "Jake", "Matthews"), _f(8, "Tahir", "Abdullayev"), _f(9, "Aljamain", "Sterling"),
          _f(10, "Kevin", "Vallejos")]


def test_parse_event_page():
    bouts = P.parse_event_page(EVENT_HTML)
    assert len(bouts) == 5
    b = bouts[0]
    assert b.red.name == "Jiří Procházka" and b.red.athlete_slug == "jiri-prochazka"
    assert b.blue.name == "Navajo Stirling" and b.blue.athlete_slug == "navajo-stirling"
    assert b.blue.image_src.endswith("STIRLING_NAVAJO_R_08-01.png?itok=aL1Vfoli")


def test_right_pose_and_full_size_url():
    src = _IMG + "2026-07/STIRLING_NAVAJO_R_08-01.png?itok=aL1Vfoli"
    assert P.is_right_pose(src)
    assert not P.is_right_pose(_IMG + "2026-04/PROCHAZKA_JIRI_L_04-11.png?itok=x")
    assert not P.is_right_pose(_IMG + "2026-04/DOS_REIS_L_04-11.png")
    assert not P.is_right_pose(None)
    assert P.full_size_url(src) == "https://www.ufc.com/images/2026-07/STIRLING_NAVAJO_R_08-01.png"
    assert (P.full_size_url(_IMG + "2025-01/5/ORTEGA_BRIAN_R_09-14.png?itok=c")
            == "https://www.ufc.com/images/2025-01/5/ORTEGA_BRIAN_R_09-14.png")


def test_match_fighter_by_folded_name_slug_and_filename():
    bouts = P.parse_event_page(EVENT_HTML)
    assert P.match_fighter(bouts[0].red, ROSTER).id == 1          # accents folded
    assert P.match_fighter(bouts[0].blue, ROSTER).id == 2          # given/family spans
    by_slug = P.Corner("blue", "N. Stirling", "navajo-stirling", None)
    assert P.match_fighter(by_slug, ROSTER).id == 2
    by_file = P.Corner("blue", "Stirling", None, bouts[0].blue.image_src)
    assert P.match_fighter(by_file, ROSTER).id == 2
    assert P.match_fighter(P.Corner("blue", "Nobody Here", "nobody-here", None), ROSTER) is None


def test_plan_event_stores_only_named_right_files():
    plan = P.Plan()
    P.plan_event(P.parse_event_page(EVENT_HTML), ROSTER, plan, "ufc-fight-night-november-21-2026")
    assert set(plan.store) == {2, 4, 5, 6}
    assert plan.store[2][1] == "https://www.ufc.com/images/2026-07/STIRLING_NAVAJO_R_08-01.png"
    assert len(plan.no_right_bouts) == 1 and "Jake Matthews" in plan.no_right_bouts[0]
    assert len(plan.unmatched) == 1 and "SOMEONE_ELSE" in plan.unmatched[0]


def test_plan_event_skips_stored_unless_refresh():
    roster = [r if r.id != 2 else _f(2, "Navajo", "Stirling", right="https://x/STIRLING_NAVAJO_R.png")
              for r in ROSTER]
    plan = P.Plan()
    P.plan_event(P.parse_event_page(EVENT_HTML), roster, plan, "s")
    assert 2 not in plan.store and plan.already == ["Navajo Stirling"]
    plan = P.Plan()
    P.plan_event(P.parse_event_page(EVENT_HTML), roster, plan, "s", refresh=True)
    assert 2 in plan.store


def test_plan_event_roster_fallback_for_bout_missing_from_our_card():
    card = [r for r in ROSTER if r.id != 4]                        # our card lacks Ortega
    plan = P.Plan()
    P.plan_event(P.parse_event_page(EVENT_HTML), card, plan, "s",
                 roster_lookup=lambda c: ROSTER[3] if c.name == "Brian Ortega" else None)
    assert 4 in plan.store


def test_shrink_caps_height_and_keeps_alpha():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGBA", (300, 1000), (10, 20, 30, 0)).save(buf, "PNG")
    data, mime = P.shrink(buf.getvalue(), max_height=500)
    img = Image.open(io.BytesIO(data))
    assert mime == "image/png" and img.height == 500 and img.width == 150
    assert img.convert("RGBA").getpixel((0, 0))[3] == 0


# ---------------------------------------------------------------------------
# Endpoint, on scratch SQLite
# ---------------------------------------------------------------------------

sqlite_only = pytest.mark.skipif(not settings.DATABASE_URL.startswith("sqlite"),
                                 reason="needs a scratch SQLite DATABASE_URL")


@pytest.fixture()
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.database import Base, SessionLocal, engine
    from app.models.ufc import UFCFighter
    from app.routers import ufc

    tables = [UFCFighter.__table__]
    Base.metadata.drop_all(engine, tables=tables)
    Base.metadata.create_all(engine, tables=tables)
    s = SessionLocal()
    now = dt.datetime(2026, 11, 16)
    s.add_all([
        UFCFighter(id=1, ufcstats_id="a", first_name="Navajo", last_name="Stirling",
                   image_url="https://ufc.com/STIRLING_NAVAJO_L.png", image_data=b"LEFT", image_mime="image/png",
                   image_right_url="https://www.ufc.com/images/2026-07/STIRLING_NAVAJO_R_08-01.png",
                   image_right_data=b"RIGHTPNG", image_right_mime="image/png", image_right_fetched_at=now),
        UFCFighter(id=2, ufcstats_id="b", first_name="Jake", last_name="Matthews",
                   image_url="https://ufc.com/MATTHEWS_JAKE_L.png", image_data=b"JAKE", image_mime="image/webp"),
        UFCFighter(id=3, ufcstats_id="c", first_name="No", last_name="Cache",
                   image_url="https://ufc.com/CACHE_NO_L.png"),
    ])
    s.commit()
    s.close()
    app = FastAPI()
    app.include_router(ufc.router)
    yield TestClient(app)
    Base.metadata.drop_all(engine, tables=tables)


@sqlite_only
def test_endpoint_serves_right_pose(client):
    r = client.get("/ufc/fighters/1/image?pose=right")
    assert r.status_code == 200 and r.content == b"RIGHTPNG"
    assert "right" in r.headers["etag"]
    default = client.get("/ufc/fighters/1/image")
    assert default.content == b"LEFT" and default.headers["etag"] != r.headers["etag"]


@sqlite_only
def test_endpoint_right_pose_falls_back_to_default(client):
    r = client.get("/ufc/fighters/2/image?pose=right")
    d = client.get("/ufc/fighters/2/image")
    assert r.status_code == 200 and r.content == d.content == b"JAKE"
    assert r.headers["etag"] == d.headers["etag"] and r.headers["content-type"] == "image/webp"
    redirect = client.get("/ufc/fighters/3/image?pose=right", follow_redirects=False)
    assert redirect.status_code == 302 and redirect.headers["location"] == "https://ufc.com/CACHE_NO_L.png"
    assert client.get("/ufc/fighters/999/image?pose=right").status_code == 404


@sqlite_only
def test_fighter_response_has_right_image(client):
    assert client.get("/ufc/fighters/1").json()["has_right_image"] is True
    assert client.get("/ufc/fighters/2").json()["has_right_image"] is False
    rows = {r["last_name"]: r["has_right_image"] for r in client.get("/ufc/fighters").json()}
    assert rows == {"Stirling": True, "Matthews": False, "Cache": False}
