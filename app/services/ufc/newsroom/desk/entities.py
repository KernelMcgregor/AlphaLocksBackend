"""Which UFC fighters a headline is about, matched against our own fighters table.

Only fighters who matter now are matchable: a UFC bout in the last four years or a
booking on an upcoming card. A full name always matches. A surname alone matches only
when it is distinctive: unique across every fighter we have ever stored (so a family
name like Nurmagomedov, shared by Umar, Khabib and others, never stands for one of
them), at least four letters, not an ordinary English word, and capitalised in the
headline (so "Van" the word never becomes Joshua Van). Multi-word nicknames that belong
to one fighter match too.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

#: Surnames that are also everyday words or very common names in headlines.
COMMON = {
    "van", "white", "brown", "young", "hill", "green", "king", "rose", "price", "miller", "ball",
    "bell", "wood", "walker", "hunter", "strong", "story", "post", "love", "grant", "long",
    "page", "rich", "west", "black", "gray", "grey", "day", "early", "wells", "hall", "may",
    "june", "march", "august", "best", "free", "fight", "night", "card", "main", "event", "title",
    "champion", "jones", "smith", "johnson", "williams", "davis", "moreno", "martin", "lewis",
    "allen", "turner", "parker", "carter", "jackson", "morgan", "ford", "hardy", "nunes", "silva",
    "santos", "costa", "lima", "oliveira", "souza", "pereira", "garcia", "rodriguez", "lopez", "perez",
}


def fold(s: str) -> str:
    """Accents off: "Dvalishvili" == "Dvalishvili", "Jiří" == "Jiri"."""
    return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c))


@dataclass
class Matcher:
    full: dict[str, int] = field(default_factory=dict)      # "ciryl gane" -> id
    surname: dict[str, int] = field(default_factory=dict)   # "Dvalishvili" -> id (case-sensitive)
    nick: dict[str, int] = field(default_factory=dict)      # "bon gamin" -> id
    names: dict[int, str] = field(default_factory=dict)     # id -> "Ciryl Gane"

    def match(self, text: str) -> set[int]:
        t = fold(text)
        low = f" {re.sub(r'[^a-z0-9 ]', ' ', t.lower())} "
        low = re.sub(r"\s+", " ", low)
        hits = {fid for name, fid in self.full.items() if f" {name} " in low}
        hits |= {fid for nk, fid in self.nick.items() if f" {nk} " in low}
        for word in set(re.findall(r"\b[A-Z][a-zA-Z'-]{3,}\b", t)):
            fid = self.surname.get(word)
            if fid is not None:
                hits.add(fid)
        return hits


def build(db, today: dt.date | None = None) -> Matcher:
    from sqlalchemy import or_
    from app.models.ufc import UFCEvent, UFCFight, UFCFighter

    today = today or dt.date.today()
    since = today - dt.timedelta(days=4 * 365)
    ids = set()
    for red, blue in (db.query(UFCFight.red_fighter_id, UFCFight.blue_fighter_id)
                      .join(UFCEvent, UFCEvent.id == UFCFight.event_id)
                      .filter(UFCEvent.date >= since)):
        ids.update((red, blue))
    fighters = db.query(UFCFighter).filter(UFCFighter.id.in_(ids)).all() if ids else []

    m = Matcher()
    # Surname uniqueness is judged against everyone stored, retired fighters included.
    last_counts = Counter(fold(n or "").strip() for (n,) in db.query(UFCFighter.last_name))
    nick_counts = Counter((f.nickname or "").strip().lower() for f in fighters if f.nickname)
    for f in fighters:
        first, last = fold(f.first_name or "").strip(), fold(f.last_name or "").strip()
        if not last:
            continue
        full = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", f"{first} {last}".lower())).strip()
        m.full[full] = f.id
        m.names[f.id] = f"{f.first_name} {f.last_name}".strip()
        # The last word of a compound surname ("Machado Garry" -> also "Garry") stays out:
        # only the stored surname as written is an alias.
        if (last_counts[last] == 1 and len(last) >= 4 and last.lower() not in COMMON
                and " " not in last):
            m.surname[last] = f.id
        nk = (f.nickname or "").strip().lower()
        if nk and len(nk.split()) >= 2 and not nk.startswith("the ") and nick_counts[nk] == 1:
            m.nick[re.sub(r"[^a-z0-9 ]", " ", fold(nk))] = f.id
    return m
