"""The newsroom's publish gate: numbers, quotes, the pick line, embeds. No database, no LLM."""

from app.services.ufc.newsroom import checker, claims, wire
from app.services.ufc.newsroom.facts import FactSheet, _expand, _ratios
from app.services.ufc.newsroom.schemas import Claim, NewsItem

STORY = ('Kape was asked about the champion. "He has delusional confidence, and I am going to '
         'expose it," Kape said. The champion won 49-47 on one card, per the newspaper.')


def sheet(**kw) -> FactSheet:
    base = dict(fight_id=1, context={}, stats_prompt="", system_prompt="",
                red={"id": 1, "name": "Joshua Van", "last": "Van"},
                blue={"id": 2, "name": "Manel Kape", "last": "Kape"},
                event_name="UFC 336", card_position=0, is_title=True, pick_corner="red",
                pick_name="Joshua Van", pick_last="Van", pick_prob=0.645, method="Decision",
                market_prob=0.52, meetings=[], allowed=_expand([0.645, 0.54, 26.0]) | _ratios("45/120"))
    base.update(kw)
    return FactSheet(**base)


CLAIM = Claim(id=7, fighter_id=2, tier="matchup", kind="beef", text="Kape mocked Van.",
              quote="He has delusional confidence, and I am going to expose it", speaker="Manel Kape",
              outlet="MMA Junkie", source_text=STORY)

GOOD = """# Van vs. Kape | UFC 336: Confidence Check

## Overview
Kape told MMA Junkie that "He has delusional confidence, and I am going to expose it." The model
has Van at 64.5% and a decision at 54%. Van landed 38% of his strikes against Taira.

```chin-quote
He has delusional confidence, and I am going to expose it
Manel Kape | MMA Junkie | October 2026 | https://example.com
```

**Prediction: Van by Decision.**"""


def test_clean_draft_passes():
    issues, unmatched = checker.check_code(GOOD, sheet(), [CLAIM])
    assert issues == [] and unmatched == []


def test_wrong_pick_line_fails():
    issues, _ = checker.check_code(GOOD.replace("Van by Decision", "Kape by Decision"), sheet(), [CLAIM])
    assert any("prediction line names" in i for i in issues)


def test_wrong_method_fails():
    issues, _ = checker.check_code(GOOD.replace("Van by Decision", "Van by KO/TKO"), sheet(), [CLAIM])
    assert any("method" in i for i in issues)


def test_number_from_the_news_fails():
    md = GOOD.replace("Van landed 38%", "The judges scored it 49-47, and Van landed 38%")
    issues, _ = checker.check_code(md, sheet(), [CLAIM])
    assert any("49" in i and "news" in i for i in issues)


def test_unknown_number_goes_to_the_reviewer():
    issues, unmatched = checker.check_code(GOOD.replace("38%", "71%"), sheet(), [CLAIM])
    assert issues == [] and unmatched == ["71"]


def test_altered_quote_fails():
    md = GOOD.replace('"He has delusional confidence, and I am going to expose it."',
                      '"He has delusional confidence and I will knock him out."')
    issues, _ = checker.check_code(md, sheet(), [CLAIM])
    assert any("not found in any source" in i for i in issues)


def test_misattributed_pull_quote_fails():
    issues, _ = checker.check_code(GOOD.replace("Manel Kape | MMA", "Joshua Van | MMA"), sheet(), [CLAIM])
    assert any("wrong person" in i for i in issues)


def test_unknown_embed_fails():
    md = GOOD.replace("**Prediction", "```chin-tweet\nhttps://x.com/a/status/1\n```\n\n**Prediction")
    issues, _ = checker.check_code(md, sheet(), [CLAIM])
    assert any("did not find" in i for i in issues)


def test_history_block_needs_meetings():
    md = GOOD.replace("## Overview", "```chin-history\n```\n\n## Overview")
    assert any("met before" in i for i in checker.check_code(md, sheet(), [CLAIM])[0])
    assert checker.check_code(md, sheet(meetings=[{"event": "x"}]), [CLAIM])[0] == []


def test_quote_matching_ignores_punctuation_and_keeps_verbatim_runs():
    text = '"F- him and his mom. It is what it is," Hokit said. "Camp was on point," Gane said. "We built something strong."'
    assert claims.verbatim("F— him and his mom. It is what it is.", text)
    assert claims.verbatim("We built something strong around me.", text) is None
    assert claims.verbatim("I will knock him out cold tonight.", text) is None


def test_queries_are_built_from_the_bout():
    import datetime as dt
    qs = wire.queries_for({"name": "Joshua Van", "last_fight": dt.date(2026, 9, 19), "ledger_size": 5},
                          {"name": "Manel Kape", "last_fight": None, "ledger_size": 0}, dt.date(2026, 11, 12))
    assert qs[0] == ('"Joshua Van" "Manel Kape" after:2026-07-15', "matchup")
    assert ('"Joshua Van" after:2026-09-20', "recent") in qs
    assert ('"Manel Kape" UFC', "background") in qs and ('"Joshua Van" UFC', "background") not in qs


def test_headlines_about_both_fighters_rank_first():
    import datetime as dt
    today = dt.date(2026, 11, 12)
    both = NewsItem(title="Kape warns Van of delusional confidence", link="https://a.com/1", published_at=today)
    one = NewsItem(title="Van open to Yan superfight", link="https://a.com/2", published_at=today)
    video = NewsItem(title="Full Fight | Van vs Taira", link="https://a.com/3", published_at=today)
    names = ["Joshua Van", "Manel Kape"]
    assert wire.score(both, names, "matchup", today) > wire.score(one, names, "matchup", today) > 0
    assert wire.score(video, names, "matchup", today) < 0


def test_inch_marks_are_not_quotes():
    md = GOOD.replace("Van landed 38%", 'Kape stands 5\' 5" with a 68" reach, and Van landed 38%')
    issues, _ = checker.check_code(md, sheet(allowed=sheet().allowed | {68.0}), [CLAIM])
    assert not any("not found in any source" in i for i in issues)
