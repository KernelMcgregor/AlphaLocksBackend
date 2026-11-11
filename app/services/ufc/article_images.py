"""Artwork for fight articles, picked automatically from Google Images (via SerpAPI).

Google's results for "A vs B" are mostly social posts and graphics: on a sample of 100,
~60% came from Instagram/Facebook/YouTube/TikTok/X, and much of the rest is odds cards,
tale-of-the-tape graphics and thumbnails with text burned in. So candidates go through:

1. Source  - social, betting and stock-agency domains are dropped; real photo outlets
             (UFC.com, ESPN, MMA Junkie...) score a bonus. Titles that name odds,
             predictions or picks are dropped (those are graphics).
2. Shape   - too small, or too tall/wide (phone screenshots, banners), is dropped.
3. Text    - the share of the image covered by text-like strokes (morphological
             gradient -> line-joined blobs with text proportions). Photos 0-1%, posters
             2-5%, graphics 7%+; top bouts reject posters, other bouts accept them.
4. Faces   - OpenCV's frontal-face cascade. Two faces (both fighters) score best, and
             the faces' centre becomes the focal point the page crops around
             (object-position), so a tall photo keeps the fighters in frame.

Which searches run depends on the bout: a main event, co-main or title fight searches
for faceoff and press-conference photos first (not only promotional posters); other
bouts search "A vs B", where a clean poster is fine. After a top bout has happened it
is searched again for a fight photo.

The chosen image is downloaded, re-encoded and stored (ufc_article_images), so pages
never hotlink; the next few candidates are kept for a manual swap.

    python -m app.services.ufc.article_images --dry-run          # next card, prints picks
    python -m app.services.ufc.article_images --fight <id> [--dry-run]
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

import httpx
import numpy as np

log = logging.getLogger("article_images")

SERPAPI_URL = "https://serpapi.com/search.json"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

#: Never used: social posts (thumbnails, screenshots, memes), betting/odds sites
#: (graphics), and stock agencies (watermarks).
BLOCKED = (
    "instagram.com", "facebook.com", "fb.com", "youtube.com", "ytimg.com", "tiktok.com",
    "x.com", "twitter.com", "threads.com", "threads.net", "reddit.com", "pinterest.",
    "kalshi.com", "polymarket.com", "sportytrader.com", "docsports.com", "mmamodel.ai",
    "oddsshark.com", "actionnetwork.com", "covers.com", "pickswise.com", "bettingpros.com",
    "dimers.com", "oddstrader.com", "sportsbookreview.com", "winnersandwhiners.com",
    "gettyimages.", "shutterstock.com", "alamy.com", "dreamstime.com", "istockphoto.com",
    "tapology.com", "agentmma.com", "next-fight.com",
)
#: Outlets whose images are mostly real event photography.
PREFERRED = {
    "ufc.com": 3.0, "espn.com": 2.5, "mmajunkie.usatoday.com": 2.5, "mmafighting.com": 2.5,
    "cbssports.com": 2.0, "sherdog.com": 2.0, "sports.yahoo.com": 1.5, "bloodyelbow.com": 1.5,
    "mmamania.com": 1.5, "si.com": 1.5, "theathletic.com": 1.5, "usatoday.com": 1.5,
    "talksport.com": 1.0, "sportskeeda.com": 0.5, "bjpenn.com": 0.5, "mmanews.com": 0.5,
}
#: Titles that announce a graphic rather than a photo.
GRAPHIC_WORDS = re.compile(
    r"\b(odds|prediction|predictions|picks?|betting|bet|parlay|tale of the tape|stats|"
    r"infographic|poll|ranking|rankings|preview card|watch party|live stream)\b", re.I)

MIN_W, MIN_H = 600, 380
ASPECT_MIN, ASPECT_MAX = 0.55, 2.3
#: Text-coverage ceilings, calibrated on a results page: photos measure 0-1%, posters
#: with headline lettering 2-5%, odds/announcement graphics 7%+. Top bouts want photos,
#: so posters are out; lower bouts accept a clean poster.
MAX_TEXT_TOP = 0.015
MAX_TEXT_CARD = 0.05
CANDIDATES_PER_QUERY = 8
STORE_MAX_W = 1280


@dataclass
class Candidate:
    url: str
    page: str
    domain: str
    title: str
    query: str
    rank: int
    width: int = 0
    height: int = 0
    text_frac: float = 0.0
    faces: int = 0
    focal_x: float = 0.5
    focal_y: float = 0.4
    score: float = 0.0
    reject: str | None = None
    jpeg: bytes | None = field(default=None, repr=False)


def _domain(url: str) -> str:
    return (urlparse(url).netloc or "").lower().removeprefix("www.")


def _blocked(domain: str) -> bool:
    return any(b in domain for b in BLOCKED)


def _preferred(domain: str) -> float:
    return next((v for d, v in PREFERRED.items() if domain == d or domain.endswith("." + d)), 0.0)


# ---------------------------------------------------------------------------
# Image analysis (pure; unit-tested)
# ---------------------------------------------------------------------------

def text_fraction(gray: np.ndarray) -> float:
    """Share of the image covered by text-like line blobs.

    Text is dense, high-contrast strokes in horizontal runs: the morphological gradient
    lights up stroke edges, a wide closing kernel fuses a word's letters into one blob,
    and blobs with text-line proportions (short, wide, densely filled) are counted.
    Photos (skin, cage mesh, crowds) rarely make blobs that wide and that dense.
    """
    import cv2

    h, w = gray.shape
    scale = 640 / w
    g = cv2.resize(gray, (640, max(1, int(h * scale))))
    grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    joined = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if not (7 <= ch <= 110) or cw < 2.0 * ch or cw < 30:
            continue
        # Density of the fused blob: a line of text fills its box solidly once its
        # letters are joined; a stray edge (a limb, the cage) leaves it mostly empty.
        fill = cv2.countNonZero(joined[y:y + ch, x:x + cw]) / float(cw * ch)
        if fill >= 0.55:
            area += cw * ch
    return area / float(g.shape[0] * g.shape[1])


def find_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """The subjects' faces: frontal and side-on (faceoffs are profile shots, both ways).

    Only faces at least ~8% of the image height count, so the crowd behind a faceoff
    can't hijack the crop; minNeighbors 8 keeps patterned clothing from reading as faces.
    """
    import cv2

    h, w = gray.shape
    eq = cv2.equalizeHist(gray)
    side = max(24, int(h * 0.08))
    found = []
    for name, flip in (("haarcascade_frontalface_default.xml", False),
                       ("haarcascade_profileface.xml", False), ("haarcascade_profileface.xml", True)):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        img = cv2.flip(eq, 1) if flip else eq
        for x, y, fw, fh in cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=8, minSize=(side, side)):
            found.append((int(w - x - fw) if flip else int(x), int(y), int(fw), int(fh)))
    # Heads sit in the upper two thirds of a fight photo; a "face" low in the frame is a
    # shirt print or a fist.
    found = [f for f in found if f[1] + f[3] / 2 <= h * 0.66]
    # One box per face: drop any box mostly inside a larger one.
    found.sort(key=lambda f: -f[2] * f[3])
    faces: list[tuple[int, int, int, int]] = []
    for f in found:
        fx, fy, fw, fh = f
        if all(not (fx < gx + gw and gx < fx + fw and fy < gy + gh and gy < fy + fh) for gx, gy, gw, gh in faces):
            faces.append(f)
    return faces[:4]


def focal_point(faces, w: int, h: int) -> tuple[float, float]:
    """Centre of the faces, nudged down a little so the crop holds head and shoulders."""
    if not faces:
        # No subject-sized face: hold the top of the frame, where fighters' heads are.
        return 0.5, 0.2
    xs = [x + fw / 2 for x, y, fw, fh in faces]
    ys = [y + fh / 2 + fh * 0.4 for x, y, fw, fh in faces]
    return round(min(1, max(0, sum(xs) / len(xs) / w)), 3), round(min(1, max(0, sum(ys) / len(ys) / h)), 3)


def _surname_hits(c: Candidate, surnames: list[str]) -> int:
    """How many of the two surnames the page title or file name mentions. A photo of
    one fighter from another bout ("Holland" alone) names one; this bout's art names both."""
    hay = f"{c.title} {c.url}".lower().replace("-", " ").replace("_", " ")
    return sum(1 for n in surnames if n and n.lower() in hay)


def analyse(c: Candidate, data: bytes, surnames: list[str] | None = None, max_text: float = MAX_TEXT_CARD) -> Candidate:
    """Decode, measure and score one downloaded candidate (sets c.reject on failure)."""
    import cv2
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        c.reject = "undecodable"
        return c
    c.width, c.height = img.size
    if c.width < MIN_W or c.height < MIN_H:
        c.reject = f"small {c.width}x{c.height}"
        return c
    aspect = c.width / c.height
    if not ASPECT_MIN <= aspect <= ASPECT_MAX:
        c.reject = f"aspect {aspect:.2f}"
        return c
    rgb = np.asarray(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    c.text_frac = round(text_fraction(gray), 4)
    if c.text_frac > max_text:
        c.reject = f"text {c.text_frac:.3f}"
        return c
    faces = find_faces(gray)
    c.faces = len(faces)
    # Relevance gate. Google mixes in other bouts (an old fight of one of them, a poster
    # for a different card); a wrong image is worse than none, so the page title or file
    # name must name both fighters. Faces are scored, not required: a faceoff shot from
    # behind or a tight crop can show none and still be the right picture.
    hits = _surname_hits(c, surnames or [])
    if surnames and hits < 2:
        c.reject = f"names {hits}/2"
        return c
    c.focal_x, c.focal_y = focal_point(faces, c.width, c.height)
    c.score = round(
        _preferred(c.domain)
        + {0: -2.0, 1: -0.5, 2: 2.0}.get(c.faces, 0.5)     # two fighters in frame is ideal
        + {0: 0.0, 1: 0.3, 2: 1.5}[_surname_hits(c, surnames or [])]
        + min(1.0, c.width / 1600)                         # resolution, capped
        - c.text_frac * 20                                 # light text (a logo) costs a little
        - c.rank * 0.12,                                   # Google's own ordering
        3)
    # Re-encode for storage: bounded size, no hotlinking, one format.
    if img.width > STORE_MAX_W:
        img = img.resize((STORE_MAX_W, int(img.height * STORE_MAX_W / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=84, optimize=True, progressive=True)
    c.jpeg = buf.getvalue()
    return c


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def queries_for(red: str, blue: str, top: bool, completed: bool) -> list[str]:
    base = f"{red} vs {blue}"
    if not top:
        return [base]
    qs = [f"{red} {blue} faceoff", f"{red} {blue} press conference", base]
    return ([f"{base} fight photo"] + qs) if completed else qs


def search(client: httpx.Client, key: str, query: str) -> list[Candidate]:
    resp = client.get(SERPAPI_URL, params={"engine": "google_images", "q": query, "api_key": key, "ijn": 0})
    resp.raise_for_status()
    out = []
    for rank, r in enumerate(resp.json().get("images_results", [])):
        url, page = r.get("original") or "", r.get("link") or ""
        dom = _domain(page) or _domain(url)
        title = r.get("title") or ""
        if not url or _blocked(dom) or _blocked(_domain(url)) or GRAPHIC_WORDS.search(title):
            continue
        ow, oh = r.get("original_width") or 0, r.get("original_height") or 0
        if ow and oh and (ow < MIN_W or oh < MIN_H):
            continue
        out.append(Candidate(url=url, page=page, domain=dom, title=title, query=query, rank=rank))
        if len(out) >= CANDIDATES_PER_QUERY:
            break
    return out


def pick_for(red: str, blue: str, top: bool, completed: bool, key: str) -> tuple[Candidate | None, list[Candidate]]:
    """Run the searches, download and score candidates; best first."""
    seen: set[str] = set()
    pool: list[Candidate] = []
    with httpx.Client(timeout=20, follow_redirects=True, headers={"User-Agent": UA}) as client:
        for q in queries_for(red, blue, top, completed):
            try:
                found = search(client, key, q)
            except Exception as e:  # one failed search shouldn't sink the rest
                log.warning(f"  search failed for {q!r}: {e}")
                continue
            for c in found:
                if c.url in seen:
                    continue
                seen.add(c.url)
                try:
                    r = client.get(c.url, headers={"Referer": c.page or "https://www.google.com/"})
                    r.raise_for_status()
                    if not r.headers.get("content-type", "").startswith("image"):
                        c.reject = "not an image"
                    else:
                        analyse(c, r.content, [red.split()[-1], blue.split()[-1]],
                                max_text=MAX_TEXT_TOP if top else MAX_TEXT_CARD)
                except Exception as e:
                    c.reject = f"download {type(e).__name__}"
                pool.append(c)
    ok = sorted((c for c in pool if not c.reject), key=lambda c: -c.score)
    return (ok[0] if ok else None), ok + [c for c in pool if c.reject]


# ---------------------------------------------------------------------------
# Which fights, and storage
# ---------------------------------------------------------------------------

def _is_top(fight) -> bool:
    return (fight.card_position is not None and fight.card_position <= 1) or "title" in (fight.weight_class or "").lower()


def _summary(c: Candidate) -> dict:
    d = asdict(c)
    d.pop("jpeg", None)
    return d


def update_article_images(dry_run: bool = False, fight_ids: list[int] | None = None,
                          refresh: bool = False) -> dict:
    """Pick artwork for every previewed fight on upcoming cards that has none, and
    re-pick a fight photo for top bouts of the last ten days that only have a pre-fight
    image. Each fight is searched once per phase, which keeps the search bill small."""
    from sqlalchemy.orm import joinedload

    from app.config import settings
    from app.database import SessionLocal
    from app.models.ufc import UFCArticleImage, UFCFight, UFCFightPreview

    key = getattr(settings, "SERPAPI_KEY", "")
    if not key:
        raise RuntimeError("SERPAPI_KEY is not set")
    db = SessionLocal()
    try:
        today = dt.date.today()
        q = (db.query(UFCFight)
             .options(joinedload(UFCFight.red_fighter), joinedload(UFCFight.blue_fighter))
             .join(UFCFightPreview, UFCFightPreview.fight_id == UFCFight.id))
        if fight_ids:
            q = q.filter(UFCFight.id.in_(fight_ids))
        else:
            q = q.filter(UFCFight.date >= today - dt.timedelta(days=10))
        fights = {f.id: f for f in q.all()}
        have = {r.fight_id: r for r in db.query(UFCArticleImage).filter(UFCArticleImage.fight_id.in_(list(fights)))}
        done, report = 0, []
        for f in sorted(fights.values(), key=lambda f: (f.date, f.card_position if f.card_position is not None else 99)):
            completed = f.winner_id is not None or (f.date and f.date < today)
            phase = "post" if completed else "pre"
            existing = have.get(f.id)
            if not fight_ids and not refresh:
                if completed and not _is_top(f):
                    continue                                 # past lower-tier bouts: leave as is
                if existing and (existing.phase == phase or not completed):
                    continue                                 # already has art for this phase
            red = f"{f.red_fighter.first_name} {f.red_fighter.last_name}".strip()
            blue = f"{f.blue_fighter.first_name} {f.blue_fighter.last_name}".strip()
            best, ranked = pick_for(red, blue, _is_top(f), completed, key)
            line = {"fight": f"{red} vs {blue}", "top": _is_top(f), "phase": phase,
                    "pick": _summary(best) if best else None,
                    "next": [_summary(c) for c in ranked[1:4]],
                    "rejected": sum(1 for c in ranked if c.reject)}
            report.append(line)
            if best is None:
                if existing is not None and not dry_run:
                    db.delete(existing)
                    db.commit()
                continue
            if dry_run:
                continue
            row = existing or UFCArticleImage(fight_id=f.id)
            row.image, row.content_type = best.jpeg, "image/jpeg"
            row.width, row.height = best.width, best.height
            row.focal_x, row.focal_y = best.focal_x, best.focal_y
            row.source_url, row.page_url, row.domain = best.url, best.page, best.domain
            row.query, row.score, row.phase = best.query, best.score, phase
            row.candidates = json.dumps([_summary(c) for c in ranked if not c.reject][:6])
            row.fetched_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
            db.add(row)
            db.commit()
            done += 1
        return {"updated": done, "dry_run": dry_run, "fights": report}
    finally:
        db.close()


def recompute_focal_points() -> int:
    """Re-run face detection on the stored images (no searches) and update their crop."""
    import cv2
    from PIL import Image

    from app.database import SessionLocal
    from app.models.ufc import UFCArticleImage

    db = SessionLocal()
    try:
        n = 0
        for row in db.query(UFCArticleImage).all():
            img = Image.open(io.BytesIO(row.image)).convert("RGB")
            faces = find_faces(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY))
            row.focal_x, row.focal_y = focal_point(faces, img.width, img.height)
            n += 1
        db.commit()
        return n
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ids = [int(sys.argv[sys.argv.index("--fight") + 1])] if "--fight" in sys.argv else None
    out = update_article_images(dry_run="--dry-run" in sys.argv, fight_ids=ids, refresh="--refresh" in sys.argv)
    for f in out["fights"]:
        p = f["pick"]
        print(f"\n{f['fight']}  ({'top' if f['top'] else 'card'}, {f['phase']}, {f['rejected']} rejected)")
        if p:
            print(f"  PICK {p['domain']:<22} score {p['score']:5.2f} faces {p['faces']} text {p['text_frac']:.3f} "
                  f"focal ({p['focal_x']},{p['focal_y']}) {p['width']}x{p['height']}\n       {p['url'][:110]}")
        for n in f["next"]:
            print(f"  next {n['domain']:<22} score {n['score']:5.2f} faces {n['faces']} text {n['text_frac']:.3f}  {n['url'][:80]}")
    print(f"\nupdated {out['updated']} (dry_run={out['dry_run']})")
