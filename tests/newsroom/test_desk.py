"""The news desk's discovery: fighter matching, clustering, picking. No database, no LLM."""

import datetime as dt

from app.services.ufc.newsroom.desk import entities, labeler, store, topics
from app.services.ufc.newsroom.schemas import NewsItem

TODAY = dt.date(2026, 11, 13)


def matcher():
    m = entities.Matcher()
    for fid, full in ((1, "merab dvalishvili"), (2, "joshua van"), (3, "umar nurmagomedov"),
                      (4, "tommy mcmillen"), (5, "ciryl gane")):
        m.full[full] = fid
        m.names[fid] = full.title()
    m.surname.update({"Dvalishvili": 1, "McMillen": 4, "Gane": 5})  # "Van" is a common word: no alias
    m.nick["bon gamin"] = 5
    return m


def test_full_names_surnames_and_nicknames():
    m = matcher()
    assert m.match("Usman Nurmagomedov sentenced over attack on Merab Dvalishvili") == {1}
    assert m.match("Dvalishvili issues statement") == {1}
    assert m.match("Bon Gamin calls Aspinall fragile") == {5}
    assert m.match("Joshua Van to defend title") == {2}


def test_van_the_word_is_never_joshua_van():
    m = matcher()
    assert m.match("Delivery van crashes after Strickland stunt") == set()
    assert m.match("Van defends belt") == set()  # surname alone is not an alias for "Van"


def test_accents_fold():
    m = entities.Matcher(full={"jiri prochazka": 9}, names={9: "Jiri Prochazka"})
    assert m.match("Jiří Procházka addresses the risk") == {9}


def item(title, outlet, day=TODAY):
    return NewsItem(title=title, outlet=outlet, link=f"https://x/{hash(title)}", published_at=day)


def test_clusters_keep_stories_apart_and_noise_out():
    m = matcher()
    items = topics.clean([
        item("Usman Nurmagomedov sentenced to six months over attack on Merab Dvalishvili", "CBS Sports"),
        item("Merab Dvalishvili issues statement after Usman Nurmagomedov attack", "Yahoo Sports"),
        item("Dvalishvili released, says he will not press charges after attack", "MMA Fighting"),
        item("Tommy McMillen says he is ready to fight days after apparent overdose", "Los Angeles Times"),
        item("Tommy McMillen releases first statement since reported overdose", "MMA Junkie"),
        item("UFC 334 predictions and best bets for Gane vs Hokit", "SI"),
        item("How to watch UFC 334 live stream", "CBS Sports"),
    ])
    assert len(items) == 5  # predictions and how-to-watch dropped
    cl = topics.cluster(items, [m.match(i.title) for i in items], today=TODAY)
    groups = sorted(sorted(i.outlet for i in c.items) for c in cl)
    assert ["CBS Sports", "MMA Fighting", "Yahoo Sports"] in groups
    assert ["Los Angeles Times", "MMA Junkie"] in groups


def test_spinoffs_fold_only_into_stories_that_get_written():
    cl = [topics.Cluster(items=[item(f"story {i}", f"o{i}")], fighter_ids=[i], outlets=[f"o{i}"],
                         newest=TODAY, score=10 - i) for i in range(4)]
    labels = [
        {"id": 0, "title": "brawl", "kind": "incident", "newsworthy": True, "merge_into": None, "follow_up": None},
        {"id": 1, "title": "pundit on brawl", "kind": "other", "newsworthy": False, "merge_into": 0, "follow_up": None},
        {"id": 2, "title": "fight week", "kind": "preview", "newsworthy": False, "merge_into": None, "follow_up": None},
        {"id": 3, "title": "retraction", "kind": "controversy", "newsworthy": True, "merge_into": 2, "follow_up": None},
    ]
    picks = labeler.pick(cl, labels, top=3)
    titles = [lab["title"] for _, lab in picks]
    assert titles == ["brawl", "retraction"]          # preview skipped, retraction not lost with it
    assert len(picks[0][0].items) == 2                # the pundit's take joined the brawl


def test_slugs():
    assert store.slugify("Josh Hokit tells Gane: 'Your career is over'", TODAY) == \
        "josh-hokit-tells-gane-your-career-is-over-20261113"


def test_a_different_first_name_is_a_different_person():
    from app.services.ufc.newsroom.claims import same_person
    fighters = [{"id": 1, "name": "Merab Dvalishvili"}, {"id": 3, "name": "Umar Nurmagomedov"}]
    assert same_person("Usman Nurmagomedov", fighters) is None
    assert same_person("Umar Nurmagomedov", fighters) == 3
    assert same_person("Dvalishvili", fighters) == 1


def test_each_development_is_its_own_story(monkeypatch):
    heads = ["Usman Nurmagomedov sentenced to six months", "Usman jailed in Abu Dhabi",
             "Merab calls for Usman's release", "Merab Dvalishvili asks Abu Dhabi to free Usman",
             "Usman sentencing: six months"]
    cl = topics.Cluster(items=[item(h, f"o{i}") for i, h in enumerate(heads)], fighter_ids=[1, 2],
                        outlets=[f"o{i}" for i in range(5)], newest=TODAY, score=10)
    reply = {"stories": [
        {"candidate": 0, "headlines": [0, 1], "title": "Usman jailed", "kind": "incident", "newsworthy": True,
         "running": None, "merge_into": None},
        {"candidate": 0, "headlines": [2, 3], "title": "Merab asks for release", "kind": "press_conference",
         "newsworthy": True, "running": "R7", "merge_into": None},
    ]}
    monkeypatch.setattr(labeler.llm, "chat_json", lambda *a, **k: reply)
    devs, labs = labeler.label(None, [cl], {1: "Usman Nurmagomedov", 2: "Merab Dvalishvili"}, [(7, "Merab asks")])
    assert [lab["title"] for lab in labs] == ["Usman jailed", "Merab asks for release"]
    assert len(devs[0].items) == 3                    # the unplaced headline went with the first
    assert [i.title for i in devs[1].items] == heads[2:4]
    assert labs[0]["running"] is None and labs[1]["running"] == 7
    assert devs[1].score < devs[0].score
