"""Training camps: ufc.com `trains_at` and Sherdog `association` resolved to one registry.

The two sources spell the same gym a dozen ways ("American Top Team", "American Top Team
ATT Coconut Creek, FL", "Jackson Wink MMA Academy - NM" vs "Jackson-Wink MMA"). Each
string is reduced to a key — accents and parentheticals dropped, a trailing location
(", FL", " - Las Vegas") cut, punctuation folded, filler words (MMA, Team, Gym, ...)
removed — then a few hand aliases apply, and a key that extends a shorter known camp key
token by token ("american top att coconut creek" -> "american top") folds into it.

A fighter can list several gyms ("Xtreme Couture / Overcome Academy"), and both sources
can disagree; every resolved camp is a membership, and the primary one is the first gym
ufc.com lists (Sherdog's when ufc.com has none). Display only — never a model input.

    python -m app.services.ufc.fighter_extras_camps --dry-run OUT.json   # prod reads, no writes
    python -m app.services.ufc.fighter_extras_camps --write              # rewrites the camp tables

Set FIGHTER_EXTRAS_DB_URL=sqlite:///<path> to point --write at a scratch database.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.ufc import SherdogFighter, UFCFight, UFCFighter, UFCFighterRanking
from app.models.ufc_fighter_extras import UFCCamp, UFCCampAlias, UFCFighterCamp
from app.services.ufc.fighter_extras_common import (
    completed_filter, division_label, division_map, extras_engine, extras_session, is_finish, outcome,
)

log = logging.getLogger("fighter_extras_camps")

#: Words that distinguish nothing between gyms.
FILLER = {"mma", "the", "team", "academy", "gym", "fight", "club", "training", "center", "centre",
          "fitness", "and", "fc", "inc", "llc", "official"}
#: Not a camp.
JUNK = {"", "freelance", "independent", "unattached", "none", "n a", "na", "unknown", "self trained",
        "no team", "various", "free agent"}
#: Spellings the rules cannot reconcile: key -> key.
MANUAL = {
    "aka": "american kickboxing",
    "aka san jose": "american kickboxing",
    "american kickboxing san jose": "american kickboxing",
    "att": "american top",
    "jacksonwink": "jackson wink",
    "jackson s": "jackson wink",
    "greg jackson s": "jackson wink",
    "alpha male urijah faber s ultimate": "alpha male",
    "urijah faber s ultimate": "alpha male",
    "nova uniao kimura": "nova uniao",
    "teixeira bethel": "teixeira",
    "fusion xcel": "fusion x cel",
    "fusion x cel performance": "fusion x cel",
    "4oz": "4oz",
    "hard knocks 365": "kill cliff",
    "sanford": "kill cliff",
    "chute boxe diego lima": "chute boxe",
    "khabib nurmagomedov": "nurmagomedov school",
    "nurmagomedov": "nurmagomedov school",
}
#: A key needs this many tokens to absorb longer keys by prefix. One word is not enough:
#: "american" would swallow both American Top Team and American Kickboxing Academy, and
#: "gracie" every Gracie affiliate.
PREFIX_MIN_TOKENS = 2


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def display_name(raw: str) -> str:
    """The gym name with its location suffix and parenthetical removed."""
    s = re.sub(r"\(.*?\)", " ", raw)
    s = re.split(r"\s+-\s+|,|\s\|\s", s)[0]
    return re.sub(r"\s+", " ", s).strip(" -.")


def camp_key(raw: str | None) -> str | None:
    """Normalised key for one gym string; None for junk."""
    if not raw:
        return None
    paren = re.findall(r"\((.*?)\)", raw)
    base = display_name(raw)
    # "AKA (American Kickboxing Academy)": an acronym before the parenthesis means the
    # parenthesis is the real name.
    if paren and re.fullmatch(r"[A-Z]{2,5}", base.split()[0] if base.split() else ""):
        base = paren[0]
    s = _fold(base).lower().replace("'", " ").replace("’", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split() if t not in FILLER]
    key = " ".join(toks) or s.strip()
    if key in JUNK:
        return None
    return MANUAL.get(key, key)


def split_gyms(raw: str | None) -> list[str]:
    """A listing that names several gyms ("Xtreme Couture / Overcome Academy")."""
    if not raw:
        return []
    return [p.strip() for p in re.split(r"\s+/\s+|;|\s+\+\s+", raw) if p.strip()]


def resolve(entries: list[tuple[int, str, str]]) -> dict:
    """entries: (fighter_id, source, raw). Returns the registry as plain data:
    {"camps": {key: {"name", "aliases": Counter(raw)}}, "members": [(fid, key, source, primary)]}"""
    keyed = []
    for fid, source, raw in entries:
        for i, part in enumerate(split_gyms(raw)):
            k = camp_key(part)
            if k:
                keyed.append((fid, source, part, k, i))

    fighters_per_key = defaultdict(set)
    for fid, _, _, k, _ in keyed:
        fighters_per_key[k].add(fid)

    # Fold longer keys into the shortest known prefix key (token-wise).
    canon = {}
    for k in sorted(fighters_per_key, key=lambda k: (len(k.split()), -len(fighters_per_key[k]))):
        toks = k.split()
        target = k
        for n in range(1, len(toks)):
            p = " ".join(toks[:n])
            if n >= PREFIX_MIN_TOKENS and p in canon and canon[p] == p:
                target = p
                break
        canon[k] = target

    camps: dict[str, dict] = {}
    members: dict[tuple[int, str], dict] = {}
    for fid, source, raw, k, idx in keyed:
        c = canon[k]
        camp = camps.setdefault(c, {"aliases": Counter(), "keys": Counter(), "names": Counter()})
        camp["aliases"][raw] += 1
        camp["keys"][k] += 1
        camp["names"][display_name(raw)] += 2 if k == c else 1
        m = members.setdefault((fid, c), {"source": source, "rank": (0 if source == "ufc" else 1, idx)})
        if (0 if source == "ufc" else 1, idx) < m["rank"]:
            m.update(source=source, rank=(0 if source == "ufc" else 1, idx))

    best = {}
    for (fid, c), m in members.items():
        if fid not in best or m["rank"] < members[(fid, best[fid])]["rank"]:
            best[fid] = c
    for c, camp in camps.items():
        # The spelling seen most (canonical-key spellings weighted double), shortest on ties.
        camp["name"] = sorted(camp["names"].items(), key=lambda kv: (-kv[1], len(kv[0])))[0][0]
    return {
        "camps": camps,
        "members": [(fid, c, m["source"], best[fid] == c) for (fid, c), m in members.items()],
    }


def slugify(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", _fold(name).lower()).strip("-") or "camp"
    slug, i = base, 2
    while slug in taken:
        slug, i = f"{base}-{i}", i + 1
    taken.add(slug)
    return slug


def gather(db: Session) -> list[tuple[int, str, str]]:
    entries = [(int(f), "ufc", t) for f, t in
               db.query(UFCFighter.id, UFCFighter.trains_at).filter(UFCFighter.trains_at.isnot(None)).all()]
    entries += [(int(f), "sherdog", a) for f, a in
                db.query(SherdogFighter.ufc_fighter_id, SherdogFighter.association)
                .filter(SherdogFighter.ufc_fighter_id.isnot(None), SherdogFighter.association.isnot(None)).all()]
    return entries


def build_rows(reg: dict) -> dict:
    """Registry -> table rows with explicit ids (stable order: biggest camp first)."""
    size = Counter(c for _, c, _, _ in reg["members"])
    order = sorted(reg["camps"], key=lambda c: (-size[c], reg["camps"][c]["name"].lower()))
    taken: set[str] = set()
    camps, aliases, members = [], [], []
    ids = {}
    for i, c in enumerate(order, 1):
        ids[c] = i
        camps.append({"id": i, "slug": slugify(reg["camps"][c]["name"], taken),
                      "name": reg["camps"][c]["name"], "n_fighters": size[c]})
    alias_rows: dict[str, dict] = {}
    for c in order:
        for raw, n in reg["camps"][c]["aliases"].items():
            k = camp_key(raw)
            row = alias_rows.setdefault(k, {"alias_key": k, "camp_id": ids[c], "raw": raw, "n_seen": 0, "_best": 0})
            row["n_seen"] += n
            if n > row["_best"]:
                row.update(raw=raw, _best=n)
    for i, row in enumerate(alias_rows.values(), 1):
        row.pop("_best")
        aliases.append({"id": i, **row})
    for i, (fid, c, source, primary) in enumerate(sorted(reg["members"]), 1):
        members.append({"id": i, "fighter_id": fid, "camp_id": ids[c], "source": source, "is_primary": primary})
    return {"camps": camps, "aliases": aliases, "members": members}


def compute(db: Session) -> dict:
    return build_rows(resolve(gather(db)))


def write(db: Session, rows: dict) -> dict:
    """Replace the three camp tables in one transaction."""
    from app.database import engine
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with extras_session(db) as xs:
        eng = extras_engine(engine)
        for t in (UFCCamp, UFCCampAlias, UFCFighterCamp):
            t.__table__.create(bind=eng, checkfirst=True)
        xs.query(UFCFighterCamp).delete()
        xs.query(UFCCampAlias).delete()
        xs.query(UFCCamp).delete()
        xs.flush()
        xs.add_all([UFCCamp(computed_at=now, **r) for r in rows["camps"]])
        xs.flush()
        xs.add_all([UFCCampAlias(**r) for r in rows["aliases"]])
        xs.add_all([UFCFighterCamp(**r) for r in rows["members"]])
        xs.commit()
    return {k: len(v) for k, v in rows.items()}


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

ACTIVE_DAYS = 730


def _ufc_records(db: Session) -> dict[int, dict]:
    """fighter -> UFC W/L/D/NC, finishes and last bout date, over every completed bout."""
    recs: dict[int, dict] = defaultdict(lambda: {"W": 0, "L": 0, "D": 0, "NC": 0, "finishes": 0, "last": None})
    rows = (db.query(UFCFight.red_fighter_id, UFCFight.blue_fighter_id, UFCFight.winner_id,
                     UFCFight.method, UFCFight.date)
            .filter(*completed_filter()).all())
    from types import SimpleNamespace
    for red, blue, winner, method, day in rows:
        f = SimpleNamespace(winner_id=winner, method=method)
        for me in (red, blue):
            if me is None:
                continue
            r = recs[int(me)]
            res = outcome(f, int(me))
            r[res] += 1
            if res == "W" and is_finish(method):
                r["finishes"] += 1
            if r["last"] is None or day > r["last"]:
                r["last"] = day
    return recs


def _ranks(db: Session) -> dict[int, int]:
    rows = (db.query(UFCFighterRanking.fighter_id, UFCFighterRanking.rank, UFCFighterRanking.weight_class)
            .filter(UFCFighterRanking.rank > 0).all())
    return {int(f): r for f, r, wc in rows if not wc.startswith("p4p")}


def _is_active(status: str | None, last) -> bool:
    if status:
        return status == "Active"
    return bool(last and last >= date.today() - timedelta(days=ACTIVE_DAYS))


def _context(db: Session) -> dict:
    """Everything the camp views share, read once."""
    with extras_session(db) as xs:
        camps = {int(c.id): c for c in xs.query(UFCCamp).all()}
        members = [(int(m.fighter_id), int(m.camp_id), m.source, m.is_primary) for m in xs.query(UFCFighterCamp).all()]
        aliases = defaultdict(list)
        for a in xs.query(UFCCampAlias).order_by(UFCCampAlias.n_seen.desc()).all():
            aliases[int(a.camp_id)].append(a.raw)
        for c in camps.values():
            xs.expunge(c)
    ids = {fid for fid, *_ in members}
    fighters = {}
    if ids:
        cols = (UFCFighter.id, UFCFighter.first_name, UFCFighter.last_name, UFCFighter.country_code,
                UFCFighter.status, UFCFighter.image_url)
        ids_l = list(ids)
        for chunk in range(0, len(ids_l), 2000):
            for row in db.query(*cols).filter(UFCFighter.id.in_(ids_l[chunk:chunk + 2000])).all():
                fighters[int(row[0])] = row
    return {"camps": camps, "members": members, "aliases": aliases, "fighters": fighters,
            "records": _ufc_records(db), "ranks": _ranks(db), "divisions": division_map(db)}


def _fighter_row(ctx: dict, fid: int, source: str | None = None, primary: bool | None = None) -> dict | None:
    f = ctx["fighters"].get(fid)
    if f is None:
        return None
    rec = ctx["records"].get(fid, {"W": 0, "L": 0, "D": 0, "NC": 0, "finishes": 0, "last": None})
    div = ctx["divisions"].get(fid)
    out = {
        "id": str(fid), "name": f"{f.first_name} {f.last_name}".strip(), "country_code": f.country_code,
        "has_image": bool(f.image_url),
        "wins": rec["W"], "losses": rec["L"], "draws": rec["D"],
        "division": div, "division_label": division_label(div),
        "rank": ctx["ranks"].get(fid),
        "active": _is_active(f.status, rec["last"]),
        "last_fight": rec["last"].isoformat() if rec["last"] else None,
    }
    if source is not None:
        out.update(source=source, primary=primary)
    return out


def _camp_summary(ctx: dict, camp, rows: list[dict]) -> dict:
    w = sum(r["wins"] for r in rows)
    lo = sum(r["losses"] for r in rows)
    fin = sum(ctx["records"].get(int(r["id"]), {}).get("finishes", 0) for r in rows)
    ranked = sorted((r for r in rows if r["rank"] and r["rank"] <= 15), key=lambda r: r["rank"])
    return {
        "id": str(camp.id), "slug": camp.slug, "name": camp.name,
        "fighters": len(rows),
        "active": sum(r["active"] for r in rows),
        "ranked": len(ranked),
        "wins": w, "losses": lo, "draws": sum(r["draws"] for r in rows),
        "win_pct": round(100 * w / (w + lo), 1) if w + lo else None,
        "finish_pct": round(100 * fin / w, 1) if w else None,
        "notable": [{"id": r["id"], "name": r["name"], "rank": r["rank"], "division_label": r["division_label"]}
                    for r in ranked[:3]],
    }


def _members_by_camp(ctx: dict) -> dict[int, list]:
    by = defaultdict(list)
    for fid, cid, source, primary in ctx["members"]:
        by[cid].append((fid, source, primary))
    return by


def camps_leaderboard(db: Session, min_fighters: int = 3) -> dict:
    ctx = _context(db)
    out = []
    for cid, mem in _members_by_camp(ctx).items():
        camp = ctx["camps"].get(cid)
        rows = [r for r in (_fighter_row(ctx, fid) for fid, _, _ in mem) if r]
        if camp is None or len(rows) < min_fighters:
            continue
        out.append(_camp_summary(ctx, camp, rows))
    out.sort(key=lambda c: (-c["active"], -c["fighters"], c["name"].lower()))
    return {"camps": out, "min_fighters": min_fighters}


def _find_camp(ctx: dict, ref: str):
    if ref.isdigit() and int(ref) in ctx["camps"]:
        return ctx["camps"][int(ref)]
    return next((c for c in ctx["camps"].values() if c.slug == ref), None)


def camp_detail(db: Session, ref: str) -> dict:
    ctx = _context(db)
    camp = _find_camp(ctx, ref)
    if camp is None:
        raise HTTPException(status_code=404, detail="Camp not found")
    mem = _members_by_camp(ctx).get(int(camp.id), [])
    rows = [r for r in (_fighter_row(ctx, fid, s, p) for fid, s, p in mem) if r]
    rows.sort(key=lambda r: (not r["active"], r["rank"] or 999, -(r["wins"]), r["name"]))
    return {**_camp_summary(ctx, camp, rows), "aliases": ctx["aliases"].get(int(camp.id), [])[:8], "roster": rows}


def fighter_camp(db: Session, fighter_id: int, limit: int = 12) -> dict:
    """The fighter's primary camp, other camps, and teammates from the primary camp."""
    with extras_session(db) as xs:
        mine = xs.query(UFCFighterCamp).filter(UFCFighterCamp.fighter_id == fighter_id).all()
        if not mine:
            return {"fighter_id": str(fighter_id), "camp": None, "other_camps": [], "teammates": []}
        primary = next((m for m in mine if m.is_primary), mine[0])
        camps = {int(c.id): c for c in xs.query(UFCCamp).filter(UFCCamp.id.in_([m.camp_id for m in mine])).all()}
        mates = [int(m.fighter_id) for m in
                 xs.query(UFCFighterCamp).filter(UFCFighterCamp.camp_id == primary.camp_id,
                                                 UFCFighterCamp.fighter_id != fighter_id).all()]
        ref = {cid: {"id": str(c.id), "slug": c.slug, "name": c.name, "fighters": c.n_fighters}
               for cid, c in camps.items()}
    ctx = {"fighters": {}, "records": _ufc_records(db), "ranks": _ranks(db), "divisions": division_map(db)}
    if mates:
        cols = (UFCFighter.id, UFCFighter.first_name, UFCFighter.last_name, UFCFighter.country_code,
                UFCFighter.status, UFCFighter.image_url)
        ctx["fighters"] = {int(r[0]): r for r in db.query(*cols).filter(UFCFighter.id.in_(mates)).all()}
    rows = [r for r in (_fighter_row(ctx, fid) for fid in mates) if r]
    rows.sort(key=lambda r: (not r["active"], r["rank"] or 999, -r["wins"], r["name"]))
    return {
        "fighter_id": str(fighter_id),
        "camp": ref.get(int(primary.camp_id)),
        "other_camps": [v for k, v in ref.items() if k != int(primary.camp_id)],
        "teammates": rows[:limit],
        "teammate_count": len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", metavar="OUT.json", help="compute from prod reads, write JSON only")
    g.add_argument("--write", action="store_true", help="replace ufc_camps / ufc_camp_aliases / ufc_fighter_camps")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        rows = compute(db)
        log.info(f"{len(rows['camps'])} camps, {len(rows['aliases'])} aliases, {len(rows['members'])} memberships")
        if args.dry_run:
            names = {c["id"]: c["name"] for c in rows["camps"]}
            by_camp = defaultdict(list)
            for a in rows["aliases"]:
                by_camp[a["camp_id"]].append(a["raw"])
            report = [{**c, "aliases": by_camp[c["id"]]} for c in rows["camps"]]
            with open(args.dry_run, "w") as fh:
                json.dump({"camps": report, "members": rows["members"], "names": names}, fh, indent=1, default=str)
            log.info(f"dry run: wrote {args.dry_run}, no database writes")
        else:
            log.info(f"wrote {write(db, rows)}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
