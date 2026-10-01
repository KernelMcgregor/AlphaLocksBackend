"""A title or interim-title bout is not a division change (fight_context_service)."""
import pytest

from app.services.ufc.fight_context_service import _division_key, _division_label


@pytest.mark.parametrize("current, previous, changed", [
    ("UFC Flyweight Title Bout", "Flyweight Bout", False),
    ("UFC Interim Flyweight Title Bout", "UFC Flyweight Title Bout", False),
    ("Flyweight Bout", "UFC Interim Flyweight Title Bout", False),
    ("UFC Light Heavyweight Title Bout", "Light Heavyweight Bout", False),
    ("Bantamweight Bout", "Flyweight Bout", True),
    ("Light Heavyweight Bout", "Heavyweight Bout", True),
    ("Women's Flyweight Bout", "Flyweight Bout", True),
    ("Catch Weight Bout", "Flyweight Bout", True),
])
def test_division_change(current, previous, changed):
    assert (_division_key(current) != _division_key(previous)) is changed


def test_label_drops_title_words():
    assert _division_label("UFC Interim Flyweight Title Bout") == "Flyweight"
    assert _division_label("Women's Strawweight Bout") == "Women's Strawweight"
