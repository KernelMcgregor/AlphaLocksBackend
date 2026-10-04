"""Parser tests for app.services.ufc.bfo_props stat markets ("X has more ...")."""

import pytest

from app.services.ufc import bfo_props as bp


def _row(label: str, fighter: int, price: str, book: int = 20, matchup: int = 42750) -> str:
    return (f'<tr class="pr"><th scope="row">{label}</th><td></td>'
            f'<td class="but-sgp" data-li="[{book},1,{matchup},170,{fighter}]"><span>{price}</span></td></tr>')


@pytest.mark.parametrize("label,fighter,key", [
    ("Burns has more significant strikes", 1, "msig_a"),
    ("Burns has more significant strikes in round 1", 2, "msigr1_b"),
    ("Burns has more takedowns", 2, "mtd_b"),
    ("Emmett lands no takedowns", 1, "tdz_a_yes"),
    ("Emmett lands at least one takedown", 1, "tdz_a_no"),
    ("Burns wins Knockout of the Night", 1, None),
])
def test_stat_labels(label, fighter, key):
    assert bp._label_key(label, fighter) == key


def test_more_markets_devig_to_corners():
    html = ('<table class="odds-table">' + _row("Burns has more takedowns", 1, "+110")
            + _row("Brady has more takedowns", 2, "-140") + "</table>")
    cons = bp.consensus(bp.parse_props(html))[42750]
    assert cons["mtd_a_nv"] + cons["mtd_b_nv"] == pytest.approx(1.0)
    # BFO fighter_a is the DB blue corner
    out = bp.corner_markets(cons, swapped=True)
    assert out["more_td_blue"]["prob"] == pytest.approx(cons["mtd_a_nv"])
    assert out["more_td_red"]["prob"] == pytest.approx(cons["mtd_b_nv"])
    assert out["more_td_blue"]["best_american"] == 110
