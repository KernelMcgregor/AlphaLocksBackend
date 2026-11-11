"""The admin dashboard's article editor: covers, framing, in-article pictures, manual
edits and context-driven rewrites.

Images arrive base64 in JSON (no multipart dependency), are re-encoded to JPEG at most
1600px wide, and get a starting focal point from face detection that the editor can
then move. Rewrites return a draft; nothing is published until the editor saves.
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
import re

from sqlalchemy.orm import Session

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
STORE_MAX_W = 1600


def _now():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def _decode_image(data_b64: str) -> tuple[bytes, int, int, float, float]:
    """base64 (optionally a data: URL) -> stored JPEG, size and a face-based focal point."""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    from app.services.ufc.article_images import find_faces, focal_point

    if "," in data_b64[:100]:
        data_b64 = data_b64.split(",", 1)[1]
    raw = base64.b64decode(data_b64)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Image is larger than 12 MB")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception as e:
        raise ValueError("Not a readable image") from e
    if img.width > STORE_MAX_W:
        img = img.resize((STORE_MAX_W, int(img.height * STORE_MAX_W / img.width)), Image.LANCZOS)
    faces = find_faces(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY))
    fx, fy = focal_point(faces, img.width, img.height)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=86, optimize=True, progressive=True)
    return buf.getvalue(), img.width, img.height, fx, fy


def _clamp(v: float) -> float:
    return round(min(1.0, max(0.0, float(v))), 4)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def list_articles(db: Session) -> list[dict]:
    """Articles for cards from a week ago onwards, soonest card first, main event first."""
    from app.models.ufc import UFCArticleImage, UFCEvent, UFCFight, UFCFightPreview

    since = dt.date.today() - dt.timedelta(days=7)
    rows = (db.query(UFCFightPreview, UFCFight, UFCEvent)
            .join(UFCFight, UFCFight.id == UFCFightPreview.fight_id)
            .join(UFCEvent, UFCEvent.id == UFCFight.event_id)
            .filter(UFCEvent.date >= since)
            .order_by(UFCEvent.date, UFCFight.card_position.is_(None), UFCFight.card_position)
            .all())
    covers = {fid for (fid,) in db.query(UFCArticleImage.fight_id)
              .filter(UFCArticleImage.fight_id.in_([f.id for _, f, _ in rows]))} if rows else set()
    out = []
    for p, f, e in rows:
        m = re.search(r"^#\s+(.+)$", p.content or "", re.M)
        out.append({
            "fight_id": str(f.id), "event_id": str(e.id), "event_name": e.name, "event_date": e.date.isoformat(),
            "card_position": f.card_position,
            "red": f"{f.red_fighter.first_name} {f.red_fighter.last_name}",
            "blue": f"{f.blue_fighter.first_name} {f.blue_fighter.last_name}",
            "headline": m.group(1).strip() if m else None,
            "has_cover": f.id in covers,
            "edited_at": p.edited_at.isoformat() if p.edited_at else None,
            "has_notes": bool(p.editor_notes),
        })
    return out


def get_article(db: Session, fight_id: int) -> dict:
    from app.models.ufc import UFCArticleImage, UFCArticleMedia, UFCFight, UFCFightPreview

    f = db.get(UFCFight, fight_id)
    p = db.query(UFCFightPreview).filter(UFCFightPreview.fight_id == fight_id).first()
    if f is None or p is None:
        raise LookupError("No article for this fight")
    cover = db.query(UFCArticleImage).filter(UFCArticleImage.fight_id == fight_id).first()
    media = db.query(UFCArticleMedia).filter(UFCArticleMedia.fight_id == fight_id).order_by(UFCArticleMedia.id).all()
    return {
        "fight_id": str(fight_id),
        "red": f"{f.red_fighter.first_name} {f.red_fighter.last_name}",
        "blue": f"{f.blue_fighter.first_name} {f.blue_fighter.last_name}",
        "event_name": f.event.name if f.event else None,
        "content": p.content,
        "editor_notes": p.editor_notes or "",
        "edited_at": p.edited_at.isoformat() if p.edited_at else None,
        "cover": None if cover is None else {
            "width": cover.width, "height": cover.height, "focal_x": cover.focal_x, "focal_y": cover.focal_y,
            "domain": cover.domain, "phase": cover.phase,
            "v": int(cover.fetched_at.timestamp()) if cover.fetched_at else 0,
            "candidates": [{"url": c["url"], "page": c.get("page"), "domain": c.get("domain"),
                            "width": c.get("width"), "height": c.get("height")}
                           for c in json.loads(cover.candidates or "[]")],
        },
        "media": [{"id": m.id, "width": m.width, "height": m.height, "focal_x": m.focal_x,
                   "focal_y": m.focal_y, "label": m.label} for m in media],
    }


# ---------------------------------------------------------------------------
# Cover
# ---------------------------------------------------------------------------

def _cover_row(db: Session, fight_id: int):
    from app.models.ufc import UFCArticleImage

    return db.query(UFCArticleImage).filter(UFCArticleImage.fight_id == fight_id).first() \
        or UFCArticleImage(fight_id=fight_id, candidates="[]")


def upload_cover(db: Session, fight_id: int, data_b64: str) -> None:
    jpeg, w, h, fx, fy = _decode_image(data_b64)
    row = _cover_row(db, fight_id)
    row.image, row.content_type, row.width, row.height = jpeg, "image/jpeg", w, h
    row.focal_x, row.focal_y = fx, fy
    row.source_url, row.page_url, row.domain, row.query = "upload", None, "upload", None
    row.score, row.phase, row.fetched_at = None, "manual", _now()
    db.add(row)
    db.commit()


def use_candidate(db: Session, fight_id: int, url: str) -> None:
    """Swap the cover for one of the stored search candidates (downloaded now)."""
    import httpx

    from app.services.ufc.article_images import UA

    row = _cover_row(db, fight_id)
    cand = next((c for c in json.loads(row.candidates or "[]") if c["url"] == url), None)
    if cand is None:
        raise LookupError("Not one of this article's candidates")
    r = httpx.get(url, headers={"User-Agent": UA, "Referer": cand.get("page") or "https://www.google.com/"},
                  timeout=25, follow_redirects=True)
    r.raise_for_status()
    jpeg, w, h, fx, fy = _decode_image(base64.b64encode(r.content).decode())
    row.image, row.content_type, row.width, row.height = jpeg, "image/jpeg", w, h
    row.focal_x, row.focal_y = fx, fy
    row.source_url, row.page_url, row.domain = url, cand.get("page"), cand.get("domain")
    row.phase, row.fetched_at = "manual", _now()
    db.commit()


def frame_cover(db: Session, fight_id: int, focal_x: float, focal_y: float) -> None:
    from app.models.ufc import UFCArticleImage

    row = db.query(UFCArticleImage).filter(UFCArticleImage.fight_id == fight_id).first()
    if row is None:
        raise LookupError("No cover to frame")
    row.focal_x, row.focal_y = _clamp(focal_x), _clamp(focal_y)
    # A new version stamp, so pages holding the old URL fetch the same bytes under the
    # new framing (the crop is applied client-side from the payload, but this also
    # refreshes the cache-busted URL everywhere).
    row.fetched_at = _now()
    db.commit()


def remove_cover(db: Session, fight_id: int) -> None:
    from app.models.ufc import UFCArticleImage

    db.query(UFCArticleImage).filter(UFCArticleImage.fight_id == fight_id).delete()
    db.commit()


# ---------------------------------------------------------------------------
# In-article pictures
# ---------------------------------------------------------------------------

def add_media(db: Session, fight_id: int, data_b64: str, label: str | None = None) -> int:
    from app.models.ufc import UFCArticleMedia

    jpeg, w, h, fx, fy = _decode_image(data_b64)
    m = UFCArticleMedia(fight_id=fight_id, image=jpeg, content_type="image/jpeg", width=w, height=h,
                        focal_x=fx, focal_y=fy, label=(label or "")[:200] or None)
    db.add(m)
    db.commit()
    return m.id


def frame_media(db: Session, media_id: int, focal_x: float, focal_y: float) -> None:
    from app.models.ufc import UFCArticleMedia

    m = db.get(UFCArticleMedia, media_id)
    if m is None:
        raise LookupError("No such picture")
    m.focal_x, m.focal_y = _clamp(focal_x), _clamp(focal_y)
    db.commit()


def delete_media(db: Session, media_id: int) -> None:
    from app.models.ufc import UFCArticleMedia

    db.query(UFCArticleMedia).filter(UFCArticleMedia.id == media_id).delete()
    db.commit()


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def save_article(db: Session, fight_id: int, content: str, editor_notes: str | None) -> None:
    from app.models.ufc import UFCFightPreview

    p = db.query(UFCFightPreview).filter(UFCFightPreview.fight_id == fight_id).first()
    if p is None:
        raise LookupError("No article for this fight")
    if not (content or "").strip():
        raise ValueError("Article is empty")
    p.content = content
    p.editor_notes = (editor_notes or "").strip() or None
    p.edited_at = _now()
    db.commit()


def rewrite_draft(db: Session, fight_id: int, notes: str, current: str | None = None) -> str:
    """Rewrite the article with the editor's context folded in. Returns the draft only."""
    from app.config import settings
    from app.models.ufc import UFCFightPreview
    from app.services.ufc.preview_service import (
        _preview_client, build_preview_prompt, gather_fight_context,
    )

    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    p = db.query(UFCFightPreview).filter(UFCFightPreview.fight_id == fight_id).first()
    context = gather_fight_context(fight_id, db)
    if p is None or context is None:
        raise LookupError("No article or fight data for this fight")
    system, user = build_preview_prompt(context)
    system += """

You are now revising an existing article. The editor has supplied background (story,
history, drama, quotes, injuries, stakes). Weave it in naturally where it belongs, usually
the opening and the fighter sections; let it shape the narrative and headline if it is the
bigger story. Treat the editor's facts as true, but do not invent quotes or events beyond
them. Keep the data-backed analysis, the section structure and the final prediction line
(the pick must not change). Keep every ```chin-image``` block exactly as written and in
roughly the same place. Return the full article in markdown, nothing else."""
    user += f"""

**EDITOR'S CONTEXT:**
{notes.strip()}

**CURRENT ARTICLE:**
{(current if current is not None else p.content).strip()}"""
    resp = _preview_client().chat.completions.create(
        model=settings.PREVIEW_MODEL, max_tokens=4096,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    out = (resp.choices[0].message.content or "").strip()
    # Some completions arrive wrapped in a ```markdown fence.
    out = re.sub(r"^```(?:markdown|md)?\s*\n([\s\S]*?)\n```$", r"\1", out)
    if not out:
        raise RuntimeError("The rewrite came back empty")
    return out
