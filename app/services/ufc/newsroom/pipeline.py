"""Run the newsroom for a fight or for the top of every upcoming card.

    python -m app.services.ufc.newsroom --upcoming             # nightly: top 5 of cards <=10 days out
    python -m app.services.ufc.newsroom --upcoming --days 90   # every announced card
    python -m app.services.ufc.newsroom --fight 1184411916109512955 --dry-run --out /tmp/nr
    python -m app.services.ufc.newsroom --ensure-schema        # create the newsroom tables

Nightly runs are cheap when nothing happened: the wire search is free, stories already
read are not re-read, and an article is only rewritten when the pick or method changed or
new matchup/recent reporting arrived.

--dry-run reads the database and calls DeepSeek but writes nothing: claims, articles and
the checker's report go to --out as JSON and markdown.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.services.ufc.newsroom import checker, claims as claims_mod, facts, llm, wire, writer
from app.services.ufc.newsroom.schemas import Article, Claim

log = logging.getLogger(__name__)

TABLES = ("ufc_news_articles", "ufc_research_claims", "ufc_article_runs", "ufc_newsroom_spend")
DATA_EDGE_PTS = 8.0  # model vs market gap at which the numbers are the story
MAX_REVISIONS = 2


def ensure_schema():
    """Create the newsroom's own tables if missing. Touches no existing table."""
    from app.database import Base, engine
    import app.models.ufc  # noqa: F401  registers the models
    tables = [t for t in Base.metadata.sorted_tables if t.name in TABLES]
    Base.metadata.create_all(engine, tables=tables)
    return [t.name for t in tables]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def newsroom_fight_ids(db: Session, days: int | None = None, top_n: int | None = None) -> list[int]:
    from app.models.ufc import UFCEvent, UFCFight, UFCFightPrediction
    days = settings.NEWSROOM_DAYS_AHEAD if days is None else days
    top_n = settings.NEWSROOM_TOP_N if top_n is None else top_n
    today = dt.date.today()
    rows = (db.query(UFCFight.id)
            .join(UFCEvent, UFCEvent.id == UFCFight.event_id)
            .join(UFCFightPrediction, UFCFightPrediction.fight_id == UFCFight.id)
            .filter(UFCFight.winner_id.is_(None), UFCFight.method.is_(None),
                    UFCEvent.date >= today, UFCEvent.date <= today + dt.timedelta(days=days),
                    UFCFight.card_position.isnot(None), UFCFight.card_position < top_n)
            .order_by(UFCEvent.date, UFCFight.card_position).all())
    return [r[0] for r in rows]


def _last_fight(db: Session, fighter_id: int) -> dt.date | None:
    from app.models.ufc import UFCFight
    return (db.query(func.max(UFCFight.date))
            .filter(or_(UFCFight.red_fighter_id == fighter_id, UFCFight.blue_fighter_id == fighter_id),
                    UFCFight.method.isnot(None)).scalar())


# ---------------------------------------------------------------------------
# Claims: stored ones, and new ones from the wire
# ---------------------------------------------------------------------------

def _to_claim(row, text: str | None) -> Claim:
    return Claim(id=row.id, fighter_id=row.fighter_id, opponent_id=row.opponent_id, tier=row.tier,
                 kind=row.kind, text=row.text, quote=row.quote, speaker=row.speaker, outlet=row.outlet,
                 show=row.show, reporter=row.reporter, venue=row.venue, source_url=row.source_url,
                 published_at=row.published_at, embed_url=row.embed_url,
                 newsworthiness=row.newsworthiness, source_text=text)


def stored_claims(db: Session, fighter_ids: list[int]) -> list[Claim]:
    from app.models.ufc import UFCNewsArticle, UFCResearchClaim
    rows = (db.query(UFCResearchClaim, UFCNewsArticle.text)
            .outerjoin(UFCNewsArticle, UFCNewsArticle.id == UFCResearchClaim.article_id)
            .filter(UFCResearchClaim.fighter_id.in_(fighter_ids), UFCResearchClaim.struck.is_(False))
            .all())
    return [_to_claim(r, t) for r, t in rows]


def gather_claims(db: Session | None, meter: llm.Meter, fs: facts.FactSheet, persist: bool,
                  log_lines: list[str]) -> list[Claim]:
    """Wire search -> new stories read and extracted -> stored (when persisting) ->
    every usable claim for both fighters, with tiers relative to this bout."""
    from app.models.ufc import UFCNewsArticle, UFCResearchClaim

    red = dict(fs.red, last_fight=_last_fight(db, fs.red["id"]))
    blue = dict(fs.blue, last_fight=_last_fight(db, fs.blue["id"]))
    known = stored_claims(db, [red["id"], blue["id"]]) if persist else []
    for f in (red, blue):
        f["ledger_size"] = sum(1 for c in known if c.fighter_id == f["id"])

    found = wire.gather(red, blue)
    stories = {}
    for item, tier, url in found:
        stories.setdefault(url, (item, tier))

    # Stories already read for this pair are not paid for twice.
    done_urls, cached = set(), {}
    if persist and stories:
        for a in db.query(UFCNewsArticle).filter(UFCNewsArticle.url.in_(list(stories))).all():
            pair = set(json.loads(a.extracted_for or "[]"))
            if {red["id"], blue["id"]} <= pair or not a.fetch_ok:
                done_urls.add(a.url)
            else:
                cached[a.url] = a
    todo = [(u, it, t) for u, (it, t) in stories.items() if u not in done_urls]

    def work(args):
        url, item, tier = args
        art = None
        if url in cached and cached[url].text:
            a = cached[url]
            art = Article(url=a.url, outlet=a.outlet, title=a.title, author=a.author,
                          published_at=a.published_at, text=a.text, db_id=a.id,
                          **json.loads(a.embeds or "{}"))
        else:
            art = wire.fetch(url, item)
        if art is None:
            return url, item, tier, None, []
        try:
            cs = claims_mod.extract(meter, art, red, blue, fs.event_name, tier, fs.fight_id)
        except llm.BudgetExceeded:
            raise
        except Exception as e:
            log.warning(f"Extraction failed for {url}: {e}")
            cs = []
        # Embeds the page carried attach to the claims that came from it.
        embed = (art.videos or art.tweets or [None])[0]
        for c in cs:
            if embed and c.kind in ("quote", "beef") and not c.embed_url:
                c.embed_url = embed
        return url, item, tier, art, cs

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(work, todo))

    new_claims: list[Claim] = []
    for url, item, tier, art, cs in results:
        log_lines.append(f"[{tier:10}] {'ok ' if art else 'n/a'} {len(cs)} claims | {item.outlet} | {item.title[:90]}")
        if persist:
            row = cached.get(url) or db.query(UFCNewsArticle).filter(UFCNewsArticle.url == url).first()
            if row is None:
                row = UFCNewsArticle(url=url[:1000], outlet=(item.outlet or "")[:200], title=item.title[:500],
                                     published_at=item.published_at, fetch_ok=art is not None)
                db.add(row)
            if art is not None:
                row.text, row.author = art.text, (art.author or None) and art.author[:300]
                row.embeds = json.dumps({"tweets": art.tweets, "videos": art.videos})
                row.fetch_ok = True
            row.extracted_for = json.dumps(sorted(set(json.loads(row.extracted_for or "[]")) | {red["id"], blue["id"]}))
            db.flush()
            for c in cs:
                dup = db.query(UFCResearchClaim.id).filter(
                    UFCResearchClaim.fighter_id == c.fighter_id, UFCResearchClaim.source_url == c.source_url,
                    UFCResearchClaim.text == c.text).first()
                if dup:
                    continue
                r = UFCResearchClaim(fight_id=fs.fight_id, article_id=row.id,
                                     **c.model_dump(exclude={"id", "source_text"}))
                db.add(r)
                db.flush()
                c.id = r.id
            db.commit()
        new_claims.extend(cs)

    if persist:
        pool_claims = stored_claims(db, [red["id"], blue["id"]])
    else:
        pool_claims = new_claims
        for i, c in enumerate(pool_claims, 1):
            c.id = i

    # Tiers relative to this bout: a claim gathered for an earlier fight is background
    # unless it is newer than the fighter's last fight.
    out = []
    for c in pool_claims:
        f = red if c.fighter_id == red["id"] else blue
        other = blue if f is red else red
        if c.opponent_id == other["id"] and (c.published_at is None or f["last_fight"] is None
                                             or c.published_at > f["last_fight"]):
            tier = "matchup"
        elif c.published_at and f["last_fight"] and c.published_at <= f["last_fight"]:
            tier = "background"
        else:
            tier = "recent" if c.tier != "background" else "background"
        out.append(c.model_copy(update={"tier": tier}))
    return _dedupe(out)


def _dedupe(cs: list[Claim]) -> list[Claim]:
    """One claim per (fighter, text start, quote): the same story syndicated by several
    outlets otherwise crowds the editor's list."""
    seen, out = set(), []
    for c in sorted(cs, key=lambda c: (-c.newsworthiness, -(c.published_at or dt.date.min).toordinal())):
        key = (c.fighter_id, (c.quote or "")[:40].lower() or c.text[:50].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def select_for_editor(cs: list[Claim], fs: facts.FactSheet) -> list[Claim]:
    """Matchup and recent reporting first; a little background per fighter."""
    rank = lambda c: (-c.newsworthiness, -(c.published_at or dt.date.min).toordinal())
    seen = set()

    def fresh(c):  # a matchup claim is stored once per fighter; the editor needs it once
        key = (c.quote or c.text)[:60].lower()
        if key in seen:
            return False
        seen.add(key)
        return True

    front = [c for c in sorted([c for c in cs if c.tier in ("matchup", "recent")], key=rank) if fresh(c)][:16]
    back = []
    for fid in (fs.red["id"], fs.blue["id"]):
        back += [c for c in sorted([c for c in cs if c.tier == "background" and c.fighter_id == fid], key=rank)
                 if fresh(c)][:3]
    return front + back


def choose_layout(fs: facts.FactSheet, cs: list[Claim], plan) -> str:
    if fs.edge_pts is not None and fs.edge_pts >= DATA_EDGE_PTS:
        return "data"
    lead = {c.id: c for c in cs}
    beef = [lead[i] for i in plan.lead_claims + plan.pull_quotes if i in lead
            and lead[i].kind in ("beef", "history") and lead[i].tier == "matchup" and lead[i].newsworthiness >= 4]
    if fs.meetings or beef:
        return "rivalry"
    if (fs.card_position is not None and fs.card_position <= 1) or fs.is_title:
        return "feature"
    return "magazine"


def strip_internal(md: str) -> str:
    return md.strip()


# ---------------------------------------------------------------------------
# One fight
# ---------------------------------------------------------------------------

def run_fight(fight_id: int, meter: llm.Meter, persist: bool, force: bool = False,
              out_dir: Path | None = None) -> dict:
    from app.models.ufc import UFCArticleRun, UFCFightPreview

    db = SessionLocal()
    log_lines: list[str] = []
    try:
        fs = facts.build(db, fight_id)
        if fs is None:
            return {"fight_id": fight_id, "status": "skipped", "reason": "no prediction or fight data"}
        name = f"{fs.red['last']} vs {fs.blue['last']}"
        preview = db.query(UFCFightPreview).filter(UFCFightPreview.fight_id == fight_id).first()
        run = db.query(UFCArticleRun).filter(UFCArticleRun.fight_id == fight_id).first() if persist else None

        cs = gather_claims(db, meter, fs, persist, log_lines)
        fresh = [c for c in cs if c.tier in ("matchup", "recent") and c.newsworthiness >= 3]
        newest = max((c.id for c in cs), default=0)
        if run and not force and run.pick == fs.pick_key and (run.max_claim_id or 0) >= newest \
                and run.status in ("published", "needs_review", "no_news"):
            return {"fight_id": fight_id, "fight": name, "status": "unchanged"}
        if not fresh:
            status = "no_news"
            result = {"fight_id": fight_id, "fight": name, "status": status, "claims": len(cs)}
            if persist:
                _save_run(db, run, fight_id, status, None, {"wire": log_lines}, fs, newest, 0.0)
            _dump(out_dir, fight_id, name, result, log_lines, cs, None, None)
            return result

        before = meter.run_usd
        picked = select_for_editor(cs, fs)
        plan = writer.plan(meter, fs, picked)
        draft = writer.write(meter, fs, picked, plan)
        report = checker.check(meter, draft, fs, picked)
        first_issues = list(report.issues)
        for _ in range(MAX_REVISIONS):
            if report.ok:
                break
            draft = writer.revise(meter, fs, picked, plan, draft, report.issues)
            report = checker.check(meter, draft, fs, picked)
        layout = choose_layout(fs, picked, plan)
        status = "published" if report.ok else "needs_review"
        usd = meter.run_usd - before
        rep = {"plan": plan.model_dump(), "issues": report.issues, "first_draft_issues": first_issues,
               "review": report.review, "claims_used": [c.id for c in picked], "wire": log_lines,
               "thesis": writer.thesis(fs), "layout": layout}

        if persist:
            if status == "published":
                if preview is not None and preview.edited_at is not None:
                    status = "skipped_edited"
                elif preview is None:
                    preview = UFCFightPreview(fight_id=fight_id, content=draft, model_used=f"{settings.PREVIEW_MODEL}+newsroom")
                    db.add(preview)
                else:
                    preview.content, preview.model_used = draft, f"{settings.PREVIEW_MODEL}+newsroom"
            elif preview is None:
                # Nothing published yet: the stats-only preview goes up meanwhile.
                from app.services.ufc.preview_service import generate_preview
                generate_preview(fight_id, db)
            _save_run(db, run, fight_id, status, layout, rep, fs, newest, usd)
            from app.services import response_cache
            response_cache.invalidate()

        result = {"fight_id": fight_id, "fight": name, "status": status, "layout": layout,
                  "claims": len(cs), "used": len(picked), "issues": report.issues, "usd": round(usd, 4)}
        _dump(out_dir, fight_id, name, {**result, "report": rep}, log_lines, cs, draft, plan)
        return result
    finally:
        db.close()


def _save_run(db, run, fight_id, status, layout, rep, fs, newest, usd):
    from app.models.ufc import UFCArticleRun
    if run is None:
        run = UFCArticleRun(fight_id=fight_id, usd=0.0)
        db.add(run)
    run.status, run.layout = status, layout or run.layout
    run.report = json.dumps(rep, default=str)
    run.pick, run.max_claim_id = fs.pick_key, newest
    run.usd = (run.usd or 0.0) + usd
    db.commit()


def _dump(out_dir, fight_id, name, result, log_lines, cs, draft, plan):
    if not out_dir:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{fight_id}_{name.replace(' ', '_')}"
    (out_dir / f"{slug}.json").write_text(json.dumps(
        {**result, "claims": [c.model_dump() for c in cs]}, indent=2, default=str))
    if draft:
        (out_dir / f"{slug}.md").write_text(draft)


# ---------------------------------------------------------------------------
# Many fights
# ---------------------------------------------------------------------------

def run(fight_ids: list[int], persist: bool, force: bool = False, out_dir: Path | None = None) -> dict:
    db = SessionLocal()
    try:
        spent = llm.month_spent(db) if persist else 0.0
    finally:
        db.close()
    meter = llm.Meter(persist=persist, month_spent=spent)
    results = []
    for fid in fight_ids:
        try:
            results.append(run_fight(fid, meter, persist, force, out_dir))
        except llm.BudgetExceeded as e:
            log.warning(str(e))
            results.append({"fight_id": fid, "status": "budget"})
            break
        except Exception as e:
            log.exception(f"Newsroom failed for fight {fid}")
            results.append({"fight_id": fid, "status": "failed", "error": str(e)[:300]})
        log.info(results[-1])
    summary = {"fights": len(results), "usd": round(meter.run_usd, 4),
               "month_usd": round(spent + meter.run_usd, 4), "results": results}
    if out_dir:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def run_upcoming(days: int | None = None, top_n: int | None = None, persist: bool = True,
                 force: bool = False, out_dir: Path | None = None) -> dict:
    db = SessionLocal()
    try:
        ids = newsroom_fight_ids(db, days, top_n)
    finally:
        db.close()
    log.info(f"Newsroom: {len(ids)} fights")
    return run(ids, persist, force, out_dir)
