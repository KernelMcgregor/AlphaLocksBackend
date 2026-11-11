"""Article image picker (app/services/ufc/article_images.py): filters, text and face
measures, focal point. No network, no database."""
from __future__ import annotations

import numpy as np

from app.services.ufc.article_images import (
    Candidate, _blocked, _surname_hits, focal_point, queries_for, text_fraction,
)


def test_social_and_betting_sources_are_blocked():
    for d in ("instagram.com", "m.facebook.com", "youtube.com", "x.com", "kalshi.com", "media.gettyimages.com"):
        assert _blocked(d), d
    for d in ("ufc.com", "mmafighting.com", "sports.yahoo.com"):
        assert not _blocked(d), d


def test_top_bouts_search_beyond_posters():
    assert queries_for("A B", "C D", top=False, completed=False) == ["A B vs C D"]
    qs = queries_for("A B", "C D", top=True, completed=False)
    assert qs[0].endswith("faceoff") and "press conference" in qs[1]
    assert queries_for("A B", "C D", top=True, completed=True)[0] == "A B vs C D fight photo"


def test_text_fraction_separates_graphics_from_flat_photo():
    import cv2

    photo = np.full((400, 640), 120, np.uint8)
    cv2.circle(photo, (320, 200), 120, 200, -1)                    # one large soft shape
    graphic = photo.copy()
    for k in range(6):
        cv2.putText(graphic, "UFC 334 ODDS PICKS PREVIEW", (20, 50 + k * 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 3)
    assert text_fraction(photo) < 0.01
    assert text_fraction(graphic) > 0.08


def test_focal_point_centres_on_faces():
    assert focal_point([], 1000, 600) == (0.5, 0.2)
    fx, fy = focal_point([(100, 100, 100, 100), (700, 120, 100, 100)], 1000, 600)
    assert 0.4 < fx < 0.5 and 0.25 < fy < 0.4


def test_surname_hits_from_title_or_filename():
    c = Candidate(url="https://x.com/medic-vs-holland.webp", page="", domain="", title="", query="", rank=0)
    assert _surname_hits(c, ["Medic", "Holland"]) == 2
    c.url, c.title = "https://x/holland.jpg", "Kevin Holland knocks out"
    assert _surname_hits(c, ["Medic", "Holland"]) == 1
