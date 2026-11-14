"""Headlines grouped into stories, and the stories scored.

Similarity between two headlines mixes what they say (TF-IDF cosine on the titles) with
who they are about (overlap of the fighters matched in them). Stories are formed by
average-linkage clustering on that similarity: a headline joins a story only if it is
close to the story as a whole, so a fight week's hundreds of loosely related headlines
do not chain into one blob. A few hundred headlines a run need nothing heavier.

Score: distinct outlets, with the big outlets counting extra, a bonus when a champion or
top-five fighter is involved, all decayed by age. Preview-type coverage (predictions,
odds, breakdowns, how to watch, live blogs, scorecards) is dropped before clustering:
the site's preview articles cover the fights themselves. What fighters say in fight week
(press conferences, media days, trash talk) stays in.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field

from app.services.ufc.newsroom.schemas import NewsItem

NOISE = re.compile(
    r"\b(predictions?|picks?|odds|betting|bets?|parlay|props?|how to watch|live (stream|results|updates|coverage|blog)|"
    r"start time|full (fight )?card|weigh-?ins?|ceremonial|scorecards?|tickets?|watch (live|free)|"
    r"fight night results|main card results|results:|stream|play-by-play|round scoring|faceoffs?|"
    r"highlights|full fight|free fight|live fight coverage|fight coverage|countdown|breakdown|preview|"
    r"embedded|net worth|biography|stats, pictures)\b", re.I)
#: Average similarity a headline needs with a story to join it.
LINK = 0.28
BIG = {"espn", "mma fighting", "mma junkie", "yahoo sports", "cbs sports", "sherdog", "the athletic",
       "sports illustrated", "bloody elbow", "los angeles times", "the independent", "bbc"}


@dataclass
class Cluster:
    items: list[NewsItem]
    fighter_ids: list[int]
    outlets: list[str]
    newest: dt.date | None
    score: float = 0.0
    key: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def headlines(self) -> list[str]:
        return [i.title for i in self.items]


def clean(items: list[NewsItem]) -> list[NewsItem]:
    return [i for i in items if len(i.title.split()) >= 4 and not NOISE.search(i.title)]


def cluster(items: list[NewsItem], fighters: list[set[int]], important: set[int] | None = None,
            today: dt.date | None = None) -> list[Cluster]:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    today = today or dt.date.today()
    important = important or set()
    n = len(items)
    if not n:
        return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    sim = cosine_similarity(vec.fit_transform([i.title for i in items]))

    # Who it is about: Jaccard overlap of the matched fighters.
    who = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            u = fighters[a] | fighters[b]
            if u:
                who[a, b] = who[b, a] = len(fighters[a] & fighters[b]) / len(u)
    dates = np.array([(today - i.published_at).days if i.published_at else 0 for i in items])
    close = np.abs(dates[:, None] - dates[None, :]) <= 2
    combined = np.where(close, 0.55 * sim + 0.45 * who, 0.0)
    np.fill_diagonal(combined, 1.0)

    labels = (AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                      distance_threshold=1 - LINK).fit(1 - combined).labels_
              if n > 1 else np.zeros(1, dtype=int))
    groups: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(i)

    out = []
    for idx in groups.values():
        its = [items[i] for i in idx]
        counts: dict[int, int] = {}
        for i in idx:
            for f in fighters[i]:
                counts[f] = counts.get(f, 0) + 1
        need = max(1, math.ceil(0.3 * len(idx)))
        fids = sorted((f for f, c in counts.items() if c >= need), key=lambda f: -counts[f])
        outlets = sorted({(i.outlet or "").strip() for i in its if i.outlet})
        big = {o for o in outlets if o.lower() in BIG}
        newest = max((i.published_at for i in its if i.published_at), default=None)
        age = max(0, (today - newest).days) if newest else 1
        if not fids and len(outlets) < 3:
            continue  # no fighter of ours and thin coverage: not our story
        score = (len(outlets) + 0.6 * len(big) + (1.5 if important & set(fids) else 0.0)
                 + 0.5 * math.log(len(its))) / (1 + age)
        out.append(Cluster(items=sorted(its, key=lambda i: i.published_at or dt.date.min, reverse=True),
                           fighter_ids=fids, outlets=outlets, newest=newest, score=round(score, 3),
                           key="-".join(map(str, sorted(fids)[:3])) or re.sub(r"\W+", "-", its[0].title.lower())[:40]))
    return sorted(out, key=lambda c: -c.score)
