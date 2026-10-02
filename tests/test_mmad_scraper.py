"""Parser / matching / loading tests for app.services.ufc.mmad_scraper (offline fixtures)."""

import datetime as dt
from pathlib import Path

import pytest

from app.services.ufc import mmad_scraper as mm

FIX = Path(__file__).parent / "fixtures" / "mmad"


def _page(name: str, did: int) -> mm.DecisionPage:
    return mm.parse_decision((FIX / name).read_text(encoding="utf-8"), did)


# ------------------------------------------------------------------- parsing


def test_parse_unanimous_five_rounds():
    p = _page("decision_16358.html", 16358)
    assert (p.fighter_a, p.fighter_a_id, p.fighter_b, p.fighter_b_id) == \
        ("Josh Van", 6491, "Alexandre Pantoja", 4147)
    assert p.outcome == "win" and p.decision_type == "Unanimous Decision"
    assert (p.mmad_event_id, p.date, p.referee) == (1648, dt.date(2026, 9, 19), "Herb Dean")
    assert [(j.mmad_judge_id, j.name, j.total_a, j.total_b) for j in p.judges] == [
        (318, "Derek Cleary", 48, 47), (94, "Sal D'Amato", 49, 46), (358, "Ron McCarthy", 50, 45)]
    assert p.judges[0].rounds == [(1, 9, 10), (2, 10, 9), (3, 10, 9), (4, 9, 10), (5, 10, 9)]
    for j in p.judges:  # rounds add up to the total
        assert sum(r[1] for r in j.rounds) == j.total_a and sum(r[2] for r in j.rounds) == j.total_b


def test_parse_media_and_fans():
    p = _page("decision_16358.html", 16358)
    assert p.media[0] == mm.MediaScore("Seán Sheehan", "SevereMMA.com", 49, 46, "a")
    assert all(m.pick in ("a", "b", "draw") for m in p.media)
    r1 = {f.pick: f.pct for f in p.fans if f.round == 1}
    assert r1 == {"b": 82.1, "a": 17.6, "draw": 0.3}       # fans gave R1 to Pantoja
    assert mm.FanScore(0, "48-47", "a", 47.9) in p.fans
    assert (p.fan_n, p.fan_a, p.fan_b, p.fan_draw) == (704, 564, 117, 23)


def test_parse_majority_draw_with_deduction():
    p = _page("decision_11610.html", 11610)
    assert p.outcome == "draw" and p.decision_type == "Majority Draw"
    assert [(j.total_a, j.total_b) for j in p.judges] == [(48, 46), (47, 47), (47, 47)]
    assert p.judges[0].rounds[2] == (3, 9, 9)               # point deduction round
    assert p.deductions == [mm.Deduction("a", 1, 3, "Low blow")]
    assert mm.deductions_by_round(p) == {0: (1, 0), 3: (1, 0)}
    assert any(f.round == 0 and f.pick == "draw" for f in p.fans)


def test_parse_old_card_unknown_judges():
    p = _page("decision_297.html", 297)
    assert p.date == dt.date(2001, 2, 23) and p.media == []
    assert all(j.mmad_judge_id is None and not j.total_known and not j.rounds_known
               for j in p.judges)


def test_parse_event_and_year_index():
    ev = mm.parse_event((FIX / "event_1648.html").read_text(encoding="utf-8"))
    assert ev.date == dt.date(2026, 9, 19)
    assert ev.decision_paths[0] == "decision/16358/Josh-Van-vs-Alexandre-Pantoja"
    assert len(ev.decision_paths) == 5
    refs = mm.parse_year_index((FIX / "year_2010.html").read_text(encoding="utf-8"))
    assert refs[1].mmad_event_id == 223 and refs[1].date == dt.date(2010, 12, 11)
    assert refs[1].n_decisions == 7


def test_candidate_events():
    d = dt.date(2010, 12, 11)
    refs = [mm.EventRef(1, "event/1/x", "UFC 124: St-Pierre vs. Koscheck 2", d, 7),
            mm.EventRef(2, "event/2/x", "Strikeforce: Henderson vs. Babalu 2", d, 4),
            mm.EventRef(3, "event/3/x", "Ortiz vs. Shamrock 3: The Final Chapter", d, 1),
            mm.EventRef(4, "event/4/x", "WEC 53", dt.date(2010, 12, 16), 5)]
    got = [(r.mmad_event_id, u) for r, u in mm.candidate_events(refs, {d})]
    assert got == [(1, True), (3, True)]
    got_all = {r.mmad_event_id: u for r, u in mm.candidate_events(refs, {d}, all_promotions=True)}
    assert got_all == {1: True, 2: False, 3: True, 4: False}


# -------------------------------------------------------------- verification


def _fight(**kw):
    base = dict(id=10, red_fighter_id=1, blue_fighter_id=2, winner_id=1, referee="Herb Dean",
                details="Derek Cleary 47 - 48. Sal D'amato 46 - 49. Ron McCarthy 45 - 50.")
    base.update(kw)
    return base


def test_verify_ok_and_swapped():
    p = _page("decision_16358.html", 16358)
    assert mm.verify(p, _fight(), swapped=False) == ("verified", [])
    # Van in the blue corner: still verified when the winner is blue.
    assert mm.verify(p, _fight(winner_id=2), swapped=True)[0] == "verified"


def test_verify_catches_wrong_winner_and_totals():
    p = _page("decision_16358.html", 16358)
    status, flags = mm.verify(p, _fight(winner_id=2), swapped=False)
    assert status == "review" and "winner" in flags
    bad = _fight(details="Derek Cleary 47 - 48. Sal D'amato 46 - 49. Ron McCarthy 46 - 49.")
    status, flags = mm.verify(p, bad, swapped=False)
    assert status == "review" and flags == ["totals"]


def test_verify_no_totals_is_name_only_and_draws_unordered():
    p = _page("decision_16358.html", 16358)
    assert mm.verify(p, _fight(details=None), swapped=False)[0] == "name_only"
    d = _page("decision_11610.html", 11610)
    f = _fight(winner_id=None, referee="Jason Herzog",
               details="Derek Cleary 46 - 48. Sal D'amato 47 - 47. Junichiro Kamijo 47 - 47.")
    assert mm.verify(d, f, swapped=False) == ("verified", [])


def test_ufcstats_jr_and_unnamed_judges_still_verify():
    """UFCStats writes "Andrew Hopper Jr." and sometimes omits a judge's name; neither
    may drop that judge's card (it used to, so these fights failed the totals check)."""
    p = _page("decision_16358.html", 16358)
    jr = _fight(details="Andrew Hopper Jr. 47 - 48. Sal D'amato 46 - 49. Ron McCarthy 45 - 50.")
    assert mm.verify(p, jr, swapped=False) == ("verified", [])
    unnamed = _fight(details="Derek Cleary 47 - 48. 46 - 49. 45 - 50.")
    assert mm.verify(p, unnamed, swapped=False) == ("verified", [])


def test_round_sum_mismatch_flagged():
    p = _page("decision_16358.html", 16358)
    p.judges[0].rounds[0] = (1, 10, 10)
    assert mm.verify(p, _fight(), swapped=False) == ("review", ["round_sum"])


# ------------------------------------------------------------------ loading


@pytest.fixture
def db(tmp_path):
    from sqlalchemy import BigInteger, create_engine
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.orm import sessionmaker
    from app.models.ufc import UFCEvent, UFCFight, UFCFighter

    @compiles(BigInteger, "sqlite")
    def _bigint_sqlite(type_, compiler, **kw):  # SQLite only auto-numbers INTEGER PKs
        return "INTEGER"

    eng = create_engine(f"sqlite:///{tmp_path}/t.db").execution_options(
        schema_translate_map={"ufc": None})
    for m in (UFCFighter, UFCEvent, UFCFight):
        m.__table__.create(eng)
    mm.ensure_tables(eng)
    s = sessionmaker(bind=eng)()
    names = {1: ("Josh", "Van"), 2: ("Alexandre", "Pantoja"), 3: ("Deiveson", "Figueiredo"),
             4: ("Brandon", "Moreno"), 5: ("Alonzo", "Menifield"), 6: ("Iwo", "Baraniewski")}
    for i, (fn, ln) in names.items():
        s.add(UFCFighter(id=i, ufcstats_id=f"f{i}", first_name=fn, last_name=ln,
                         wins=0, losses=0, draws=0))
    s.add(UFCEvent(id=1, ufcstats_id="e1", name="UFC 331", date=dt.date(2026, 9, 19)))
    s.add(UFCEvent(id=2, ufcstats_id="e2", name="UFC 256", date=dt.date(2020, 12, 12)))
    # Van is the BLUE corner here, to exercise the corner swap.
    s.add(UFCFight(id=10, ufcstats_id="b10", event_id=1, date=dt.date(2026, 9, 19),
                   red_fighter_id=2, blue_fighter_id=1, winner_id=1,
                   method="Decision - Unanimous", referee="Herb Dean",
                   details="Derek Cleary 47 - 48. Sal D'amato 46 - 49. Ron McCarthy 45 - 50."))
    s.add(UFCFight(id=11, ufcstats_id="b11", event_id=2, date=dt.date(2020, 12, 12),
                   red_fighter_id=3, blue_fighter_id=4, winner_id=None,
                   method="Decision - Majority", referee="Jason Herzog",
                   details="Derek Cleary 46 - 48. Sal D'amato 47 - 47. Junichiro Kamijo 47 - 47."))
    # Same night as UFC 331, no mmad page given to the loader: gets UFCStats totals only.
    s.add(UFCFight(id=12, ufcstats_id="b12", event_id=1, date=dt.date(2026, 9, 19),
                   red_fighter_id=5, blue_fighter_id=6, winner_id=5,
                   method="Decision - Split", referee="Mark Smith",
                   details="Chris Leben 28 - 29. Ron McCarthy 29 - 28. Jovany Varela 28 - 29."))
    s.commit()
    yield s
    s.close()


def test_load_pages_end_to_end(db):
    from app.models.ufc import (
        MMADDecision, MMADFanScore, MMADFighter, MMADMediaScore, UFCJudge, UFCJudgeAlias,
        UFCJudgeScorecard,
    )
    pages = [(_page("decision_16358.html", 16358), True), (_page("decision_11610.html", 11610), True)]
    res = mm.load_pages(db, pages)
    db.commit()
    assert res["stats"]["verified"] == 2 and not res["review"]

    d = db.query(MMADDecision).filter_by(mmad_decision_id=16358).one()
    assert (d.fight_id, d.swapped, d.match_status) == (10, True, "verified")
    # Cleary R1 went 10-9 to Pantoja, who is RED in the DB.
    cleary = db.query(UFCJudge).filter_by(mmad_judge_id=318).one()
    r1 = db.query(UFCJudgeScorecard).filter_by(source="mmad", fight_id=10, judge_id=cleary.id,
                                               round=1).one()
    assert (r1.red_pts, r1.blue_pts) == (10, 9)
    tot = db.query(UFCJudgeScorecard).filter_by(source="mmad", fight_id=10, judge_id=cleary.id,
                                                round=0).one()
    assert (tot.red_pts, tot.blue_pts) == (47, 48)

    # UFCStats spellings were learnt as aliases of the mmad judges.
    alias = db.query(UFCJudgeAlias).filter_by(alias_norm=mm.normalize_name("Sal D'amato")).one()
    assert db.get(UFCJudge, alias.judge_id).mmad_judge_id == 94
    # Fight 12 has only UFCStats totals: McCarthy is reused, Leben/Varela are new judges.
    rows = db.query(UFCJudgeScorecard).filter_by(source="ufcstats", fight_id=12).all()
    assert len(rows) == 3
    mcc = db.query(UFCJudge).filter_by(mmad_judge_id=358).one()
    # "29 - 28" lists the loser first: McCarthy was the dissenting judge (red won 2-1).
    assert any(r.judge_id == mcc.id and (r.red_pts, r.blue_pts) == (28, 29) for r in rows)
    assert db.query(UFCJudge).filter(UFCJudge.mmad_judge_id.is_(None)).count() == 2

    # Deductions: Figueiredo (red) lost a point in R3; the judges' own verdict was 10-9.
    from app.models.ufc import MMADDeduction
    r3 = db.query(UFCJudgeScorecard).filter_by(source="mmad", fight_id=11, round=3).all()
    assert len(r3) == 3
    assert all((r.red_pts, r.blue_pts, r.red_ded, r.blue_ded) == (9, 9, 1, 0) for r in r3)
    assert all((r.red_pts + r.red_ded, r.blue_pts + r.blue_ded) == (10, 9) for r in r3)
    assert db.query(MMADDeduction).filter_by(mmad_decision_id=11610).one().reason == "Low blow"

    links = {f.mmad_fighter_id: (f.ufc_fighter_id, f.match_status) for f in db.query(MMADFighter)}
    assert links[6491] == (1, "verified") and links[4147] == (2, "verified")
    assert db.query(MMADMediaScore).filter_by(mmad_decision_id=16358).count() == 20
    assert db.query(MMADFanScore).filter_by(mmad_decision_id=16358, round=1).count() == 3

    # Idempotent: a second load replaces rather than duplicates.
    n = db.query(UFCJudgeScorecard).count()
    mm.load_pages(db, pages)
    db.commit()
    assert db.query(UFCJudgeScorecard).count() == n
    assert db.query(UFCJudge).count() == 6


def test_fighter_conflict_left_unlinked(db):
    from app.models.ufc import MMADFighter
    p1 = _page("decision_16358.html", 16358)
    p2 = _page("decision_11610.html", 11610)
    p2.fighter_a_id = p1.fighter_a_id          # same mmad id on two different DB fighters
    mm.load_pages(db, [(p1, True), (p2, True)])
    rec = db.query(MMADFighter).filter_by(mmad_fighter_id=p1.fighter_a_id).one()
    assert rec.match_status == "conflict" and rec.ufc_fighter_id is None


def test_wrong_totals_go_to_review(db):
    from app.models.ufc import MMADDecision, UFCJudgeScorecard
    p = _page("decision_16358.html", 16358)
    p.judges[2].total_a, p.judges[2].total_b = 49, 46
    p.judges[2].rounds[3] = (4, 9, 10)
    res = mm.load_pages(db, [(p, True)])
    assert res["stats"] == {"review": 1, "ufcstats_rows": 6}
    d = db.query(MMADDecision).filter_by(mmad_decision_id=16358).one()
    assert d.match_status == "review" and d.flags == "totals" and d.fight_id == 10
    # Cards are kept for inspection but not attached to the fight.
    assert db.query(UFCJudgeScorecard).filter_by(source="mmad", fight_id=10).count() == 0


class _FakeClient:
    """Serves fixtures for the sync; counts 'network' calls."""

    def __init__(self, pages: dict[str, str]):
        self.pages, self.n_network = pages, 0

    def year_index(self, year, max_age_s=None):
        self.n_network += 1
        return [mm.EventRef(1648, "event/1648/x", "UFC 331: Van vs. Pantoja 2",
                            dt.date(2026, 9, 19), 1)]

    def event(self, path, max_age_s=None):
        self.n_network += 1
        return mm.EventPage(1648, "UFC 331", dt.date(2026, 9, 19),
                            ["decision/16358/Josh-Van-vs-Alexandre-Pantoja"])

    def decision(self, path, max_age_s=None):
        self.n_network += 1
        return mm.parse_decision(self.pages[path], mm._href_id(path, "decision"))


def test_sync_loads_pending_then_is_quiet(db):
    from app.models.ufc import MMADDecision, UFCJudgeScorecard
    client = _FakeClient({"decision/16358/Josh-Van-vs-Alexandre-Pantoja":
                          (FIX / "decision_16358.html").read_text(encoding="utf-8")})
    today = dt.date(2026, 9, 21)
    res = mm.sync(db, client, days=30, today=today)
    db.commit()
    assert res["pending"] == 2 and res["loaded"] == 1      # fights 10 and 12 were pending
    assert db.query(MMADDecision).filter_by(fight_id=10, match_status="verified").count() == 1
    assert db.query(UFCJudgeScorecard).filter_by(source="ufcstats", fight_id=12).count() == 3
    # Fight 12 is still pending (no mmad page yet), so the next run fetches again ...
    res2 = mm.sync(db, client, days=30, today=today)
    assert res2["pending"] == 1
    # ... but once nothing is pending, the sync makes no requests at all.
    db.query(UFCJudgeScorecard).filter_by(fight_id=12).delete()
    from app.models.ufc import UFCFight
    db.query(UFCFight).filter_by(id=12).delete()
    db.commit()
    before = client.n_network
    res3 = mm.sync(db, client, days=30, today=today)
    assert res3["pending"] == 0 and client.n_network == before
