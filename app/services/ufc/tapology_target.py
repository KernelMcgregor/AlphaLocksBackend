"""Tapology's published rankings, as the fit target.

Tapology keeps their scoring confidential, so the only way to land on their ordering is to
calibrate against their output. tapology.com returns HTTP 403 to automated requests and
the full lists sit behind a login, so the target is pasted in by hand rather than scraped.

Put the rankings in `data/tapology_rankings.txt` in this shape — division header, then one
fighter per line, most-highly-ranked first. Rank numbers, records and any other trailing
columns are ignored, so pasting straight off the page works:

    == lightweight
    1 Ilia Topuria
    2 Arman Tsarukyan
    3 Charles Oliveira
    ...

    == welterweight
    1 Islam Makhachev
    ...

Division headers accept either the internal key (`w_strawweight`) or the display name
("Women's Strawweight", "Light Heavyweight"). Name matching against our database is
case- and accent-insensitive; anything that cannot be matched is reported rather than
silently dropped, because a quietly unmatched fighter makes the fit look better than it is.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

from app.services.ufc.tapology_rankings import DIVISIONS

log = logging.getLogger("tapology_target")

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "tapology_rankings.txt"

#: Accepted spellings for a division header, beyond the internal key itself.
_HEADER_ALIASES = {
    "flyweight": "flyweight",
    "bantamweight": "bantamweight",
    "featherweight": "featherweight",
    "lightweight": "lightweight",
    "welterweight": "welterweight",
    "middleweight": "middleweight",
    "light heavyweight": "light_heavyweight",
    "heavyweight": "heavyweight",
    "womens strawweight": "w_strawweight",
    "womens flyweight": "w_flyweight",
    "womens bantamweight": "w_bantamweight",
}


#: Letters NFKD will not take apart, because the stroke/slash is part of the glyph rather
#: than a combining mark. Without these, "Jan Błachowicz" folds to "jan b achowicz" and
#: fails to match our "Jan Blachowicz" — a silently dropped fighter, which shrinks the
#: target and flatters the fit.
_NON_DECOMPOSING = str.maketrans({
    "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ø": "o", "Ø": "O",
    "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss",
    "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D", "ı": "i",
})


def normalise(name: str) -> str:
    """Fold a fighter name to a comparison key.

    Accents, punctuation and case all differ between Tapology's spelling and ufcstats'
    ("Jose Aldo" / "José Aldo", "Ji Yeon Kim" / "Ji-Yeon Kim"), and a fit that silently
    drops those fighters would be measuring the wrong population.
    """
    s = name.translate(_NON_DECOMPOSING)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z ]+", " ", s.lower())
    return " ".join(s.split())


def _parse_header(line: str) -> str | None:
    raw = line.lstrip("=# ").strip().rstrip(":").strip()
    if not raw:
        return None
    key = raw.lower().replace("_", " ")
    key = re.sub(r"[^a-z ]+", "", key).strip()
    key = key.replace("women s", "womens").replace("women", "womens").replace("womenss", "womens")
    key = " ".join(key.split())
    if raw.lower() in DIVISIONS:
        return raw.lower()
    return _HEADER_ALIASES.get(key)


#: A UFC record as Tapology prints it: "11-5", "12-2-1", "25-11, 1 NC".
_RECORD = re.compile(r"^\d+-\d+(-\d+)?(,\s*\d+\s*NC)?$", re.I)

#: `Justin "The Highlight" Gaethje` -> `Justin Gaethje`
_NICKNAME = re.compile(r'\s*["“‘\'].*?["”’\']\s*')


#: Header furniture that appears once, inside the first fighter's block.
_FURNITURE = {"UFC", "STRENGTH", "OF", "SCHEDULE", "HELP"}
_ARROWS = "▲▼△▽"


def _parse_page_dump(lines: list[str]) -> list[tuple[str, int | None]]:
    """Parse a raw copy-paste of a Tapology rankings page.

    Their page spreads each fighter over a dozen lines — rank, movement arrow, name, name
    again with nickname, record, "UFC", the Strength of Schedule number, then method/year
    pairs for the last six bouts.

    The anchor is the bare `UFC` line, because it is the only element present for EVERY
    fighter. The record is not: several heavyweights (Cortes-Acosta, Kuniev, Hokit, Delija,
    Spivac, Pinto, Pericić, Gaziev, Teixeira) have no record line at all, and anchoring on
    the record silently dropped all of them — which would have quietly shortened the
    heavyweight target to half its real depth and made the fit look better than it was.

    Returns (name, sos) in page order. `sos` is None when the page did not show one.
    """
    anchors = [i for i, ln in enumerate(lines) if ln.strip().upper() == "UFC"]
    out: list[tuple[str, int | None]] = []

    for idx, a in enumerate(anchors):
        # -- name: nearest line above that is not the record, a rank, or an arrow
        name = ""
        for j in range(a - 1, max(-1, a - 6), -1):
            cand = lines[j].strip()
            if (not cand or cand.isdigit() or cand in _ARROWS
                    or _RECORD.match(cand) or cand.upper() in _FURNITURE):
                continue
            name = _NICKNAME.sub(" ", cand).strip()
            break
        if not name:
            continue

        # -- everything between this anchor and the next fighter's name block
        end = anchors[idx + 1] if idx + 1 < len(anchors) else len(lines)
        body = [ln.strip() for ln in lines[a + 1:end]]

        # "Inactive for 21 Months" — Tapology shows these under a `snooze` marker as
        # NOT ranked. Including them would shift every rank below by one.
        if any("inactive for" in ln.lower() for ln in body):
            continue

        sos = None
        for tok in body:
            if not tok or tok.upper() in _FURNITURE:
                continue
            if tok.isdigit():
                val = int(tok)
                sos = val if 1 <= val <= 99 else None   # a year is not an SoS
            break
        out.append((" ".join(name.split()), sos))
    return out


def parse(text: str) -> dict[str, list[str]]:
    """Parsed target: division -> fighter names, best first."""
    return {d: [n for n, _ in rows] for d, rows in parse_full(text).items()}


def parse_full(text: str) -> dict[str, list[tuple[str, int | None]]]:
    """As `parse`, but keeping each fighter's published Strength of Schedule.

    SoS is worth capturing because it is the one number Tapology both discloses the
    formula for AND publishes the output of — which makes it a far stronger calibration
    target for the opponent-tier curve than the ordering alone.
    """
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("#") and not re.match(r"^#\d", line):
            continue

        header = _parse_header(line) if line else None
        if header:
            current = header
            blocks.setdefault(current, [])
            continue
        if current is None:
            continue
        blocks[current].append(raw_line)

    out: dict[str, list[tuple[str, int | None]]] = {}
    for division, lines in blocks.items():
        # A page dump is recognised by the presence of record lines; anything else is
        # treated as the simple one-name-per-line form.
        if any(_RECORD.match(ln.strip()) for ln in lines):
            rows = _parse_page_dump(lines)
        else:
            rows = []
            for ln in lines:
                s = ln.strip()
                if not s:
                    continue
                name = re.sub(r"^\s*#?\d+[.)]?\s+", "", s)
                name = re.split(r"\s{2,}|\t|\s+\(\d+[-–]\d+", name)[0].strip()
                name = _NICKNAME.sub(" ", name).strip()
                if name and not name.isdigit():
                    rows.append((" ".join(name.split()), None))
        if rows:
            out[division] = rows
    return out


def _read(path: Path | str | None) -> str:
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"No Tapology target at {p}.\n"
            "Paste the rankings for each division into that file — see the module "
            "docstring for the format."
        )
    return p.read_text(encoding="utf-8")


def load(path: Path | str | None = None) -> dict[str, list[str]]:
    return parse(_read(path))


def load_full(path: Path | str | None = None) -> dict[str, list[tuple[str, int | None]]]:
    return parse_full(_read(path))


def resolve(target: dict[str, list[str]], names: dict[int, str]) -> tuple[dict, list]:
    """Map target names onto fighter ids.

    Returns (division -> [fighter_id, ...], unmatched) — the unmatched list is returned
    rather than logged away so the caller has to decide what to do about it.
    """
    by_key: dict[str, list[int]] = {}
    for fid, name in names.items():
        by_key.setdefault(normalise(name), []).append(fid)

    resolved: dict[str, list[int]] = {}
    unmatched: list[tuple[str, str]] = []
    for division, fighters in target.items():
        ids: list[int] = []
        for name in fighters:
            candidates = by_key.get(normalise(name))
            if not candidates:
                unmatched.append((division, name))
                continue
            # An ambiguous name is worse than a missing one: picking the wrong Silva
            # corrupts the target silently, where a gap only shrinks it.
            if len(candidates) > 1:
                unmatched.append((division, f"{name} (ambiguous: {len(candidates)} matches)"))
                continue
            ids.append(candidates[0])
        if ids:
            resolved[division] = ids
    return resolved, unmatched
