"""Clean promotion labels for Sherdog bouts (read side only; the scraper is untouched).

sherdog_scraper.derive_promotion regexes the event-name prefix. Its known problems:
  * ONE is split three ways ('One Championship', 'One FC', 'ONE': case-sensitive pattern);
  * The Ultimate Fighter house bouts are labelled 'UFC';
  * short abbreviations lump unrelated shows together (AFC = Arena / Ambitions / All
    Fighters ..., also EFC, CFC, FFC, EC, RITC).
`clean(event_name, promotion)` fixes the first two and, for an ambiguous abbreviation,
keys the promotion on the event's own title words so different shows stay apart.
"""
from __future__ import annotations

import re

AMBIGUOUS = {"AFC", "EFC", "CFC", "FFC", "EC", "RITC", "GC", "PFC", "MFC", "SFC", "WFC", "IFC",
             "GFC", "UCE", "ROC", "WLF", "ZST"}
TUF = re.compile(r"ultimate fighter|\bTUF\b", re.I)
ONE = re.compile(r"^one\b|^one (championship|fc)\b", re.I)


def clean(event_name: str | None, promotion: str | None) -> str:
    name = (event_name or "").strip()
    promo = (promotion or "").strip()
    if TUF.search(name) and promo.upper() == "UFC" and not re.search(r"finale", name, re.I):
        return "TUF"
    if ONE.search(name) or promo.lower() in ("one championship", "one fc", "one"):
        return "ONE"
    if promo.upper() in AMBIGUOUS:
        # 'AFC 12 - Arena Fight Championship' -> 'AFC:arena fight championship'
        _, _, suffix = name.partition(" - ")
        words = re.sub(r"[\d#]+", "", suffix).strip().lower()
        return f"{promo.upper()}:{words}" if words else f"{promo.upper()}:?"
    return promo or "unknown"
