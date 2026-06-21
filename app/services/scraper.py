"""
UFC Stats scraper — fetches all fighter, event, fight, and fight stats data
from ufcstats.com and upserts into the database.

Usage:
    python -m app.services.scraper           # full scrape
    python -m app.services.scraper --update   # only new events since last in DB
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import time
from datetime import datetime

import httpx
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.ufc import UFCEvent, UFCFight, UFCFighter, UFCFightStats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "http://ufcstats.com"
REQUEST_DELAY = 0.3  # seconds between requests to be polite


# ---------------------------------------------------------------------------
# HTTP client with JS challenge solver
# ---------------------------------------------------------------------------

def _solve_pow_challenge(html: str) -> dict[str, str] | None:
    """Solve the SHA-256 proof-of-work challenge ufcstats.com uses."""
    match = re.search(r'nonce="([^"]+)"', html)
    target_match = re.search(r"target\.length\)!==target\.length\)", html)
    if not match:
        return None

    nonce = match.group(1)
    # Find target length — count zeros required
    zeros_match = re.search(r"new Array\((\d+)\+1\)", html)
    if not zeros_match:
        return None

    num_zeros = int(zeros_match.group(1))
    target = "0" * num_zeros
    n = 0
    while True:
        candidate = f"{nonce}:{n}"
        h = hashlib.sha256(candidate.encode()).hexdigest()
        if h[:num_zeros] == target:
            return {"nonce": nonce, "n": str(n)}
        n += 1


class Scraper:
    def __init__(self):
        self.client = httpx.Client(
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
            follow_redirects=True,
            timeout=30.0,
        )
        self._cookies_solved = False

    def fetch(self, url: str, retries: int = 3) -> BeautifulSoup | None:
        for attempt in range(retries):
            try:
                resp = self.client.get(url)
                # Check for JS challenge
                if "Checking your browser" in resp.text and not self._cookies_solved:
                    log.info("Solving PoW challenge...")
                    solution = _solve_pow_challenge(resp.text)
                    if solution:
                        self.client.post(
                            f"{BASE_URL}/__c",
                            data=solution,
                            headers={"Content-Type": "application/x-www-form-urlencoded"},
                        )
                        self._cookies_solved = True
                        resp = self.client.get(url)
                    else:
                        log.warning("Could not solve PoW challenge")
                        return None

                if resp.status_code == 200 and "Checking your browser" not in resp.text:
                    return BeautifulSoup(resp.text, "html.parser")

            except httpx.HTTPError as e:
                log.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(2)

        log.error(f"Failed to fetch {url} after {retries} attempts")
        return None

    def close(self):
        self.client.close()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_id(url: str) -> str:
    """Extract the hex ID from a ufcstats URL."""
    return url.rstrip("/").split("/")[-1]


def _parse_landed_attempted(text: str) -> tuple[int, int]:
    """Parse '30 of 92' into (30, 92)."""
    text = text.strip()
    if not text or text == "---":
        return 0, 0
    parts = text.split(" of ")
    if len(parts) != 2:
        return 0, 0
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        return 0, 0


def _parse_ctrl_seconds(text: str) -> int:
    """Parse '2:32' or '--' into seconds."""
    text = text.strip()
    if not text or text in ("--", "---"):
        return 0
    parts = text.split(":")
    if len(parts) != 2:
        return 0
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return 0


def _parse_time_to_seconds(time_str: str) -> int:
    """Parse '4:35' into 275 seconds."""
    time_str = time_str.strip()
    parts = time_str.split(":")
    if len(parts) != 2:
        return 0
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return 0


def _compute_fight_time(finish_round: int, finish_time_str: str, time_format: str) -> tuple[int, int]:
    """Compute total fight time and max fight time from round format.
    Returns (fight_time_seconds, max_fight_time_seconds).
    """
    if not time_format:
        return 0, 0

    round_durations = []
    for part in time_format.split("-"):
        try:
            round_durations.append(int(part.strip()) * 60)
        except ValueError:
            pass

    max_time = sum(round_durations)
    finish_seconds = _parse_time_to_seconds(finish_time_str) if finish_time_str else 0

    fight_time = sum(round_durations[: max(0, finish_round - 1)]) + finish_seconds
    return fight_time, max_time


def _clean_text(text: str | None) -> str:
    """Strip and collapse whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


# ---------------------------------------------------------------------------
# Scrape functions
# ---------------------------------------------------------------------------

def scrape_all_fighters(scraper: Scraper) -> list[dict]:
    """Scrape fighter listing pages (a-z) and individual fighter pages for DOB."""
    fighters = []
    for char in "abcdefghijklmnopqrstuvwxyz":
        url = f"{BASE_URL}/statistics/fighters?char={char}&page=all"
        log.info(f"Fetching fighters: {char}")
        soup = scraper.fetch(url)
        if not soup:
            continue

        rows = soup.select("tbody tr.b-statistics__table-row")
        for row in rows:
            cells = row.select("td")
            if len(cells) < 11:
                continue

            links = row.select("td:nth-child(1) a")
            if not links:
                continue

            link = links[0].get("href", "")
            if "fighter-details" not in link:
                continue

            first_name = _clean_text(cells[0].get_text())
            last_name = _clean_text(cells[1].get_text())
            nickname = _clean_text(cells[2].get_text())
            height = _clean_text(cells[3].get_text())
            weight = _clean_text(cells[4].get_text())
            reach = _clean_text(cells[5].get_text())
            stance = _clean_text(cells[6].get_text())
            wins = _clean_text(cells[7].get_text())
            losses = _clean_text(cells[8].get_text())
            draws = _clean_text(cells[9].get_text())

            fighter = {
                "ufcstats_id": _extract_id(link),
                "first_name": first_name,
                "last_name": last_name,
                "nickname": nickname if nickname and nickname != "--" else None,
                "height": height if height and height != "--" else None,
                "weight": weight if weight and weight != "--" else None,
                "reach": reach if reach and reach != "--" else None,
                "stance": stance if stance and stance != "--" else None,
                "wins": int(wins) if wins.isdigit() else 0,
                "losses": int(losses) if losses.isdigit() else 0,
                "draws": int(draws) if draws.isdigit() else 0,
                "link": link,
            }
            fighters.append(fighter)
        time.sleep(REQUEST_DELAY)

    # Fetch DOB from individual fighter pages
    log.info(f"Fetching DOB for {len(fighters)} fighters...")
    for i, fighter in enumerate(fighters):
        if i % 100 == 0:
            log.info(f"  DOB progress: {i}/{len(fighters)}")
        soup = scraper.fetch(fighter["link"])
        if soup:
            dob_li = soup.select_one("ul.b-list__box-list li:nth-child(5)")
            if dob_li:
                dob_text = _clean_text(dob_li.get_text())
                dob_match = re.search(r"([A-Za-z]+ \d{1,2}, \d{4})", dob_text)
                if dob_match:
                    try:
                        fighter["dob"] = datetime.strptime(dob_match.group(1), "%b %d, %Y").date()
                    except ValueError:
                        fighter["dob"] = None
                else:
                    fighter["dob"] = None
            else:
                fighter["dob"] = None
        else:
            fighter["dob"] = None
        time.sleep(REQUEST_DELAY)

    return fighters


def _scrape_fighter_listings_only(scraper: Scraper) -> list[dict]:
    """Scrape fighter listing pages (a-z) for W/L/D records only. Skips DOB."""
    fighters = []
    for char in "abcdefghijklmnopqrstuvwxyz":
        url = f"{BASE_URL}/statistics/fighters?char={char}&page=all"
        soup = scraper.fetch(url)
        if not soup:
            continue

        rows = soup.select("tbody tr.b-statistics__table-row")
        for row in rows:
            cells = row.select("td")
            if len(cells) < 11:
                continue
            links = row.select("td:nth-child(1) a")
            if not links:
                continue
            link = links[0].get("href", "")
            if "fighter-details" not in link:
                continue

            wins = _clean_text(cells[7].get_text())
            losses = _clean_text(cells[8].get_text())
            draws = _clean_text(cells[9].get_text())

            fighters.append({
                "ufcstats_id": _extract_id(link),
                "first_name": _clean_text(cells[0].get_text()),
                "last_name": _clean_text(cells[1].get_text()),
                "nickname": _clean_text(cells[2].get_text()) or None,
                "height": _clean_text(cells[3].get_text()) or None,
                "weight": _clean_text(cells[4].get_text()) or None,
                "reach": _clean_text(cells[5].get_text()) or None,
                "stance": _clean_text(cells[6].get_text()) or None,
                "wins": int(wins) if wins.isdigit() else 0,
                "losses": int(losses) if losses.isdigit() else 0,
                "draws": int(draws) if draws.isdigit() else 0,
            })
        time.sleep(REQUEST_DELAY)

    return fighters


def scrape_all_events(scraper: Scraper, since_date=None) -> list[dict]:
    """Scrape all completed events. If since_date, only return events after that date."""
    url = f"{BASE_URL}/statistics/events/completed?page=all"
    log.info("Fetching events list...")
    soup = scraper.fetch(url)
    if not soup:
        return []

    events = []
    rows = soup.select("tbody tr.b-statistics__table-row")
    for row in rows:
        link_el = row.select_one("td:nth-child(1) a")
        if not link_el:
            continue

        link = link_el.get("href", "")
        if "event-details" not in link and "eventdetails" not in link:
            continue

        name = _clean_text(link_el.get_text())
        date_el = row.select_one("td:nth-child(1) span")
        date_str = _clean_text(date_el.get_text()) if date_el else ""

        try:
            event_date = datetime.strptime(date_str, "%B %d, %Y").date()
        except ValueError:
            continue

        if since_date and event_date <= since_date:
            continue

        location_el = row.select_one("td:nth-child(2)")
        location = _clean_text(location_el.get_text()) if location_el else None

        events.append({
            "ufcstats_id": _extract_id(link),
            "name": name,
            "date": event_date,
            "location": location,
            "link": link,
        })

    log.info(f"Found {len(events)} events")
    return events


def scrape_event_fights(scraper: Scraper, event_link: str) -> list[str]:
    """Get all fight detail URLs from an event page."""
    soup = scraper.fetch(event_link)
    if not soup:
        return []

    fight_links = []
    for a in soup.select("tbody tr.b-fight-details__table-row"):
        link = a.get("data-link", "")
        if "fight-details" in link:
            fight_links.append(link)

    return fight_links


def scrape_fighter_fight_links(scraper: Scraper, fighter_link: str) -> list[dict]:
    """Get all fight detail URLs + dates from a fighter's profile page.
    Returns list of {"url": str, "date": date|None}.
    """
    soup = scraper.fetch(fighter_link)
    if not soup:
        return []

    fights = []
    table = soup.select_one("table.b-fight-details__table")
    if not table:
        return []

    for row in table.select("tbody tr.b-fight-details__table-row"):
        link = row.get("data-link", "")
        if "fight-details" not in link:
            continue

        # Date is embedded in the event column (col 6), e.g. "UFC 264... Jul. 10, 2021"
        fight_date = None
        cells = row.select("td")
        if len(cells) > 6:
            event_text = _clean_text(cells[6].get_text())
            date_match = re.search(r"([A-Za-z]{3}\.\s+\d{1,2},\s+\d{4})", event_text)
            if date_match:
                try:
                    fight_date = datetime.strptime(date_match.group(1), "%b. %d, %Y").date()
                except ValueError:
                    pass

        fights.append({"url": link, "date": fight_date})

    return fights


def scrape_fight_details(scraper: Scraper, fight_url: str) -> dict | None:
    """Scrape a single fight page for fighters, results, and stats."""
    soup = scraper.fetch(fight_url)
    if not soup:
        return None

    result = {"ufcstats_id": _extract_id(fight_url)}

    # Event info from fight page
    event_link_el = soup.select_one("h2.b-content__title a")
    if event_link_el:
        result["event_url"] = event_link_el.get("href", "")
        result["event_name"] = _clean_text(event_link_el.get_text())
        result["event_ufcstats_id"] = _extract_id(result["event_url"])

    # Fighter sections
    persons = soup.select("div.b-fight-details__person")
    if len(persons) < 2:
        return None

    for i, (key, person) in enumerate(zip(["red", "blue"], persons[:2])):
        link_el = person.select_one("h3 a")
        result_el = person.select_one("i")
        result[f"{key}_url"] = link_el.get("href", "") if link_el else ""
        result[f"{key}_name"] = _clean_text(link_el.get_text()) if link_el else ""
        result[f"{key}_result"] = _clean_text(result_el.get_text()) if result_el else ""

    # Fight info
    fight_head = soup.select_one("div.b-fight-details__fight")
    if fight_head:
        weight_class_el = fight_head.select_one("i.b-fight-details__fight-title")
        result["weight_class"] = _clean_text(weight_class_el.get_text()) if weight_class_el else None

        # Method is in i.b-fight-details__text-item_first
        method_el = fight_head.select_one("i.b-fight-details__text-item_first")
        if method_el:
            inner_is = method_el.select("i")
            if len(inner_is) >= 2:
                result["method"] = _clean_text(inner_is[1].get_text())

        # Round, Time, Time format are in i.b-fight-details__text-item
        text_items = fight_head.select("i.b-fight-details__text-item")
        for item in text_items:
            label_el = item.select_one("i.b-fight-details__label")
            if not label_el:
                continue
            label = _clean_text(label_el.get_text()).rstrip(":")
            value = _clean_text(item.get_text().replace(label_el.get_text(), ""))

            if label == "Round":
                try:
                    result["finish_round"] = int(value)
                except ValueError:
                    result["finish_round"] = None
            elif label == "Time":
                result["finish_time"] = value
            elif label == "Time format":
                fmt_match = re.search(r"\(([^)]+)\)", value)
                result["time_format"] = fmt_match.group(1) if fmt_match else value
            elif label == "Referee":
                result["referee"] = value

        # Details is a loose text node in the second <p> tag
        detail_ps = fight_head.select("p.b-fight-details__text")
        if len(detail_ps) >= 2:
            # Get all text that isn't inside <i> tags
            p_text = detail_ps[1].get_text()
            # Remove the "Details:" label text
            details_clean = re.sub(r"Details:\s*", "", p_text)
            details_clean = _clean_text(details_clean)
            if details_clean:
                result["details"] = details_clean

    # Compute fight time
    finish_round = result.get("finish_round", 0) or 0
    finish_time = result.get("finish_time", "")
    time_format = result.get("time_format", "")
    ft, mft = _compute_fight_time(finish_round, finish_time, time_format)
    result["fight_time_seconds"] = ft
    result["max_fight_time_seconds"] = mft

    result["stats"] = []

    # Page has 4 tables:
    #   Table 0: Totals (1 row — fight totals for KD, SigStr, TotalStr, TD, SubAtt, Rev, Ctrl)
    #   Table 1: Per-round version of Table 0 (N rows, one per round)
    #   Table 2: Sig strikes totals (1 row — Head, Body, Leg, Distance, Clinch, Ground)
    #   Table 3: Per-round version of Table 2 (N rows, one per round)
    all_tables = soup.select("table")
    totals_table = all_tables[0] if len(all_tables) >= 1 else None
    totals_per_round = all_tables[1] if len(all_tables) >= 2 else None
    sig_totals_table = all_tables[2] if len(all_tables) >= 3 else None
    sig_per_round = all_tables[3] if len(all_tables) >= 4 else None

    totals_fields = ["name", "kd", "sig_str", "sig_str_pct", "total_str", "td", "td_pct", "sub_att", "rev", "ctrl"]
    sig_fields = ["name", "sig_str2", "sig_str_pct2", "head", "body", "leg", "distance", "clinch", "ground"]

    def _parse_table_rows(table, fields):
        """Parse all rows of a table into list of [{fighter0}, {fighter1}] per row."""
        if not table:
            return []
        parsed = []
        for row in table.select("tbody tr"):
            cells = row.select("td")
            row_data = [{}, {}]
            for cell_idx, field in enumerate(fields):
                if cell_idx >= len(cells):
                    break
                values = cells[cell_idx].select("p")
                for fighter_idx in range(min(2, len(values))):
                    row_data[fighter_idx][field] = _clean_text(values[fighter_idx].get_text())
            parsed.append(row_data)
        return parsed

    # Totals = 1 row from table 0 + per-round rows from table 1
    totals_total = _parse_table_rows(totals_table, totals_fields)
    totals_rounds = _parse_table_rows(totals_per_round, totals_fields)
    totals_rows = totals_total + totals_rounds  # row 0 = totals, row 1+ = per round

    sig_total = _parse_table_rows(sig_totals_table, sig_fields)
    sig_rounds = _parse_table_rows(sig_per_round, sig_fields)
    sig_rows = sig_total + sig_rounds  # same structure

    num_rounds = max(len(totals_rows), len(sig_rows))

    for row_idx in range(num_rounds):
        # round_number: 0 = totals, 1+ = per round
        round_number = row_idx

        for fighter_idx, corner in enumerate(["red", "blue"]):
            t = totals_rows[row_idx][fighter_idx] if row_idx < len(totals_rows) else {}
            s = sig_rows[row_idx][fighter_idx] if row_idx < len(sig_rows) else {}

            sig_landed, sig_att = _parse_landed_attempted(t.get("sig_str", ""))
            total_landed, total_att = _parse_landed_attempted(t.get("total_str", ""))
            td_landed, td_att = _parse_landed_attempted(t.get("td", ""))
            head_l, head_a = _parse_landed_attempted(s.get("head", ""))
            body_l, body_a = _parse_landed_attempted(s.get("body", ""))
            leg_l, leg_a = _parse_landed_attempted(s.get("leg", ""))
            dist_l, dist_a = _parse_landed_attempted(s.get("distance", ""))
            clinch_l, clinch_a = _parse_landed_attempted(s.get("clinch", ""))
            ground_l, ground_a = _parse_landed_attempted(s.get("ground", ""))

            try:
                kd = int(t.get("kd", 0))
            except (ValueError, TypeError):
                kd = 0
            try:
                sub_att = int(t.get("sub_att", 0))
            except (ValueError, TypeError):
                sub_att = 0
            try:
                rev_val = int(t.get("rev", 0))
            except (ValueError, TypeError):
                rev_val = 0

            stat = {
                "round_number": round_number,
                "corner": corner,
                "fighter_ufcstats_id": _extract_id(result.get(f"{corner}_url", "")),
                "kd": kd,
                "sig_str_landed": sig_landed,
                "sig_str_attempted": sig_att,
                "total_str_landed": total_landed,
                "total_str_attempted": total_att,
                "td_landed": td_landed,
                "td_attempted": td_att,
                "sub_att": sub_att,
                "rev": rev_val,
                "ctrl_seconds": _parse_ctrl_seconds(t.get("ctrl", "")),
                "head_landed": head_l,
                "head_attempted": head_a,
                "body_landed": body_l,
                "body_attempted": body_a,
                "leg_landed": leg_l,
                "leg_attempted": leg_a,
                "distance_landed": dist_l,
                "distance_attempted": dist_a,
                "clinch_landed": clinch_l,
                "clinch_attempted": clinch_a,
                "ground_landed": ground_l,
                "ground_attempted": ground_a,
            }
            result["stats"].append(stat)

    return result


# ---------------------------------------------------------------------------
# Database upsert functions
# ---------------------------------------------------------------------------

def upsert_fighters(db: Session, fighters: list[dict]):
    """Insert or update fighters by ufcstats_id."""
    for f in fighters:
        existing = db.query(UFCFighter).filter(UFCFighter.ufcstats_id == f["ufcstats_id"]).first()
        if existing:
            for key in ("first_name", "last_name", "nickname", "height", "weight", "reach", "stance", "dob", "wins", "losses", "draws"):
                if key in f and f[key] is not None:
                    setattr(existing, key, f[key])
        else:
            fighter = UFCFighter(
                ufcstats_id=f["ufcstats_id"],
                first_name=f["first_name"],
                last_name=f["last_name"],
                nickname=f.get("nickname"),
                height=f.get("height"),
                weight=f.get("weight"),
                reach=f.get("reach"),
                stance=f.get("stance"),
                dob=f.get("dob"),
                wins=f.get("wins", 0),
                losses=f.get("losses", 0),
                draws=f.get("draws", 0),
            )
            db.add(fighter)
    db.commit()
    log.info(f"Upserted {len(fighters)} fighters")


def upsert_events(db: Session, events: list[dict]):
    """Insert or update events by ufcstats_id."""
    for e in events:
        existing = db.query(UFCEvent).filter(UFCEvent.ufcstats_id == e["ufcstats_id"]).first()
        if existing:
            existing.name = e["name"]
            existing.date = e["date"]
            existing.location = e.get("location")
        else:
            event = UFCEvent(
                ufcstats_id=e["ufcstats_id"],
                name=e["name"],
                date=e["date"],
                location=e.get("location"),
            )
            db.add(event)
    db.commit()
    log.info(f"Upserted {len(events)} events")


def upsert_event_from_fight(db: Session, fight_data: dict) -> int | None:
    """Create or find event from fight page data. Returns event DB id."""
    event_id = fight_data.get("event_ufcstats_id")
    if not event_id:
        return None

    existing = db.query(UFCEvent).filter(UFCEvent.ufcstats_id == event_id).first()
    if existing:
        return existing.id

    # Use fight date if available, otherwise leave as None-safe fallback
    fight_date = fight_data.get("date") or datetime.now().date()

    event = UFCEvent(
        ufcstats_id=event_id,
        name=fight_data.get("event_name", ""),
        date=fight_date,
    )
    db.add(event)
    db.flush()
    return event.id


def upsert_fight(db: Session, fight_data: dict, event_db_id: int) -> int | None:
    """Insert or update a single fight and its stats. Returns fight DB id."""
    ufcstats_id = fight_data["ufcstats_id"]

    # Look up fighter DB IDs
    red_id_str = _extract_id(fight_data.get("red_url", ""))
    blue_id_str = _extract_id(fight_data.get("blue_url", ""))
    red_fighter = db.query(UFCFighter).filter(UFCFighter.ufcstats_id == red_id_str).first()
    blue_fighter = db.query(UFCFighter).filter(UFCFighter.ufcstats_id == blue_id_str).first()

    if not red_fighter or not blue_fighter:
        log.warning(f"Skipping fight {ufcstats_id}: fighter not found (red={red_id_str}, blue={blue_id_str})")
        return None

    # Determine winner
    winner_id = None
    red_result = fight_data.get("red_result", "").upper()
    blue_result = fight_data.get("blue_result", "").upper()
    if red_result == "W":
        winner_id = red_fighter.id
    elif blue_result == "W":
        winner_id = blue_fighter.id

    existing = db.query(UFCFight).filter(UFCFight.ufcstats_id == ufcstats_id).first()
    if existing:
        fight = existing
        fight.event_id = event_db_id
        fight.red_fighter_id = red_fighter.id
        fight.blue_fighter_id = blue_fighter.id
        fight.winner_id = winner_id
        fight.red_result = red_result
        fight.blue_result = blue_result
        fight.date = fight_data.get("date")
        fight.weight_class = fight_data.get("weight_class")
        fight.method = fight_data.get("method")
        fight.details = fight_data.get("details")
        fight.referee = fight_data.get("referee")
        fight.finish_round = fight_data.get("finish_round")
        fight.finish_time = fight_data.get("finish_time")
        fight.time_format = fight_data.get("time_format")
        fight.fight_time_seconds = fight_data.get("fight_time_seconds")
        fight.max_fight_time_seconds = fight_data.get("max_fight_time_seconds")
    else:
        fight = UFCFight(
            ufcstats_id=ufcstats_id,
            date=fight_data.get("date"),
            event_id=event_db_id,
            red_fighter_id=red_fighter.id,
            blue_fighter_id=blue_fighter.id,
            winner_id=winner_id,
            red_result=red_result,
            blue_result=blue_result,
            weight_class=fight_data.get("weight_class"),
            method=fight_data.get("method"),
            details=fight_data.get("details"),
            referee=fight_data.get("referee"),
            finish_round=fight_data.get("finish_round"),
            finish_time=fight_data.get("finish_time"),
            time_format=fight_data.get("time_format"),
            fight_time_seconds=fight_data.get("fight_time_seconds"),
            max_fight_time_seconds=fight_data.get("max_fight_time_seconds"),
        )
        db.add(fight)
    db.flush()

    # Upsert fight stats
    fighter_map = {"red": red_fighter, "blue": blue_fighter}
    for stat_data in fight_data.get("stats", []):
        corner = stat_data["corner"]
        fighter = fighter_map.get(corner)
        if not fighter:
            continue

        round_num = stat_data.get("round_number", 0)
        existing_stat = (
            db.query(UFCFightStats)
            .filter(
                UFCFightStats.fight_id == fight.id,
                UFCFightStats.fighter_id == fighter.id,
                UFCFightStats.round_number == round_num,
            )
            .first()
        )
        if existing_stat:
            for key, val in stat_data.items():
                if key not in ("corner", "fighter_ufcstats_id"):
                    setattr(existing_stat, key, val)
        else:
            db.add(UFCFightStats(
                fight_id=fight.id,
                fighter_id=fighter.id,
                **{k: v for k, v in stat_data.items() if k != "fighter_ufcstats_id"},
            ))

    db.commit()
    return fight.id


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Upcoming events scrape
# ---------------------------------------------------------------------------

def scrape_upcoming(max_events: int = 5):
    """Scrape upcoming events and their fight matchups into ufc_fights.

    Upcoming fights get inserted with winner_id=NULL and no stats.
    When the event completes and a regular scrape runs, upsert_fight
    fills in the winner, method, stats, etc. by matching ufcstats_id.
    """
    log.info(f"Scraping upcoming events (max={max_events})")
    scraper = Scraper()
    db = SessionLocal()

    try:
        # Step 1: Get upcoming events listing
        soup = scraper.fetch(f"{BASE_URL}/statistics/events/upcoming?page=all")
        if not soup:
            log.error("Failed to fetch upcoming events page")
            return

        events = []
        for row in soup.select("tbody tr.b-statistics__table-row"):
            link_el = row.select_one("a")
            if not link_el:
                continue
            link = link_el.get("href", "")
            if "event-details" not in link:
                continue

            name = _clean_text(link_el.get_text())
            date_el = row.select_one("td:nth-child(1) span")
            date_str = _clean_text(date_el.get_text()) if date_el else ""
            try:
                event_date = datetime.strptime(date_str, "%B %d, %Y").date()
            except ValueError:
                continue

            location_el = row.select_one("td:nth-child(2)")
            location = _clean_text(location_el.get_text()) if location_el else None

            events.append({
                "ufcstats_id": _extract_id(link),
                "name": name,
                "date": event_date,
                "location": location,
                "link": link,
            })

        events = events[:max_events]
        log.info(f"Found {len(events)} upcoming events")
        upsert_events(db, events)

        # Step 2: For each event, scrape fight detail pages for matchup info
        total_fights = 0
        for event_data in events:
            log.info(f"Scraping fights for: {event_data['name']} ({event_data['date']})")

            fight_links = scrape_event_fights(scraper, event_data["link"])
            log.info(f"  Found {len(fight_links)} fights")
            time.sleep(REQUEST_DELAY)

            event_db = db.query(UFCEvent).filter(
                UFCEvent.ufcstats_id == event_data["ufcstats_id"]
            ).first()
            if not event_db:
                continue

            for fight_url in fight_links:
                fight_data = scrape_fight_details(scraper, fight_url)
                if not fight_data:
                    continue
                fight_data["date"] = event_data["date"]
                upsert_fight(db, fight_data, event_db.id)
                total_fights += 1
                time.sleep(REQUEST_DELAY)

        db.commit()
        log.info(f"Upcoming scrape complete: {len(events)} events, {total_fights} fights")

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Fast post-event update: only scrape recently completed events
# ---------------------------------------------------------------------------

def run_recent_update():
    """Fast update pipeline for the day after an event.

    1. Find the most recent completed event already in the DB
    2. Scrape the completed events page for anything newer
    3. For each new event, scrape its fights (fills in winner/stats for
       previously-upcoming fights via upsert_fight)
    4. Update fighter records for fighters involved in those events
    5. Refresh upcoming events
    6. Re-generate predictions for new fights
    """
    log.info("Starting recent update")
    scraper = Scraper()
    db = SessionLocal()

    try:
        # Step 1: Find the most recent completed event date in DB
        from sqlalchemy import func
        latest = db.query(func.max(UFCEvent.date)).filter(
            UFCFight.event_id == UFCEvent.id,
            UFCFight.winner_id.isnot(None),
        ).scalar()

        if latest:
            log.info(f"Most recent completed event in DB: {latest}")
        else:
            log.info("No completed events in DB, will scrape all")

        # Step 2: Get completed events newer than our latest
        all_completed = scrape_all_events(scraper)
        new_events = []
        for ev in all_completed:
            if latest is None or ev["date"] > latest:
                new_events.append(ev)

        if not new_events:
            log.info("No new completed events found")
        else:
            log.info(f"Found {len(new_events)} new completed events to process")
            upsert_events(db, new_events)

            # Step 3: For each new event, scrape its fights
            fighter_ids_to_update = set()
            for event_data in new_events:
                log.info(f"Scraping fights for: {event_data['name']} ({event_data['date']})")
                fight_links = scrape_event_fights(scraper, event_data["link"])
                log.info(f"  Found {len(fight_links)} fights")
                time.sleep(REQUEST_DELAY)

                event_db = db.query(UFCEvent).filter(
                    UFCEvent.ufcstats_id == event_data["ufcstats_id"]
                ).first()
                if not event_db:
                    continue

                for fight_url in fight_links:
                    fight_data = scrape_fight_details(scraper, fight_url)
                    if not fight_data:
                        continue
                    fight_data["date"] = event_data["date"]
                    upsert_fight(db, fight_data, event_db.id)

                    # Track fighters to update their records
                    red_id = _extract_id(fight_data.get("red_url", ""))
                    blue_id = _extract_id(fight_data.get("blue_url", ""))
                    if red_id:
                        fighter_ids_to_update.add(red_id)
                    if blue_id:
                        fighter_ids_to_update.add(blue_id)
                    time.sleep(REQUEST_DELAY)

            # Step 4: Update fighter records (wins/losses/draws) from listing pages
            # Only fetch the listing pages (no individual DOB pages) for speed
            log.info(f"Updating records for {len(fighter_ids_to_update)} fighters")
            fighters_from_listing = _scrape_fighter_listings_only(scraper)
            fighters_to_upsert = [
                f for f in fighters_from_listing if f["ufcstats_id"] in fighter_ids_to_update
            ]
            if fighters_to_upsert:
                upsert_fighters(db, fighters_to_upsert)
                log.info(f"Updated {len(fighters_to_upsert)} fighter records")

        # Step 5: Refresh upcoming events
        log.info("Refreshing upcoming events...")
        db.close()
        scraper.close()
        scrape_upcoming()

        log.info("Recent update complete")

    except Exception:
        log.exception("Recent update failed")
        db.rollback()
    finally:
        try:
            db.close()
        except Exception:
            pass
        try:
            scraper.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main scrape orchestration
# ---------------------------------------------------------------------------

def run_scrape(mode: str = "full"):
    """
    Scrape pipeline:
    1. Scrape all fighters from the fighters listing (a-z) + DOB from each page
    2. Scrape all events listing for date/location info
    3. For each fighter, visit their profile to get all fight links
    4. For each unique fight, scrape the fight detail page for stats

    This approach goes through fighter profiles (not events) to ensure we catch
    all fights including non-UFC events like Strikeforce, WEC, etc.
    """
    log.info(f"Starting scrape (mode={mode})")
    scraper = Scraper()
    db = SessionLocal()

    try:
        # Step 1: Scrape all fighters
        fighters = scrape_all_fighters(scraper)
        upsert_fighters(db, fighters)

        # Step 2: Scrape events listing for date/location
        events = scrape_all_events(scraper)
        upsert_events(db, events)

        # Step 3: Collect all unique fight links + dates from fighter profiles
        # Key: fight_url, Value: date from fighter profile
        all_fights_to_scrape: dict[str, object] = {}
        if mode == "update":
            existing_fights = {f.ufcstats_id for f in db.query(UFCFight.ufcstats_id).all()}
            log.info(f"Update mode: {len(existing_fights)} fights already in DB")
        else:
            existing_fights = set()

        for i, fighter in enumerate(fighters):
            if i % 200 == 0:
                log.info(f"Collecting fight links: {i}/{len(fighters)} fighters")
            fight_entries = scrape_fighter_fight_links(scraper, fighter["link"])
            for entry in fight_entries:
                fight_id = _extract_id(entry["url"])
                if mode == "update" and fight_id in existing_fights:
                    continue
                # Keep the date (first one found wins, they should all agree)
                if entry["url"] not in all_fights_to_scrape:
                    all_fights_to_scrape[entry["url"]] = entry["date"]
            time.sleep(REQUEST_DELAY)

        log.info(f"Found {len(all_fights_to_scrape)} unique fights from fighter profiles")

        # Step 4: Cross-check events for any fights missed by fighter profiles
        missed = 0
        for i, event in enumerate(events):
            if i % 100 == 0:
                log.info(f"Cross-checking events: {i}/{len(events)}")
            fight_links = scrape_event_fights(scraper, event["link"])
            for link in fight_links:
                if link not in all_fights_to_scrape:
                    fight_id = _extract_id(link)
                    if mode == "update" and fight_id in existing_fights:
                        continue
                    all_fights_to_scrape[link] = event["date"]
                    missed += 1
            time.sleep(REQUEST_DELAY)

        if missed:
            log.info(f"Found {missed} additional fights from events not on any fighter profile")
        log.info(f"Total unique fights to scrape: {len(all_fights_to_scrape)}")

        # Step 5: Scrape each fight detail page
        for i, (fight_url, fight_date) in enumerate(all_fights_to_scrape.items()):
            if i % 100 == 0:
                log.info(f"Scraping fights: {i}/{len(all_fights_to_scrape)}")

            fight_data = scrape_fight_details(scraper, fight_url)
            if not fight_data:
                continue

            # Attach date from fighter profile
            fight_data["date"] = fight_date

            # Get or create the event
            event_db_id = upsert_event_from_fight(db, fight_data)
            if not event_db_id:
                log.warning(f"Skipping fight {fight_data['ufcstats_id']}: no event info")
                continue

            upsert_fight(db, fight_data, event_db_id)
            time.sleep(REQUEST_DELAY)

        # Final counts
        log.info(
            f"Scrape complete: "
            f"{db.query(UFCFighter).count()} fighters, "
            f"{db.query(UFCEvent).count()} events, "
            f"{db.query(UFCFight).count()} fights, "
            f"{db.query(UFCFightStats).count()} fight stats"
        )

        # Populate derived stats on fight stat rows
        from app.services.fight_stats_derived_service import compute_all_derived_stats
        log.info("Computing derived fight stats...")
        n = compute_all_derived_stats()
        log.info(f"Derived stats updated on {n} fight stat rows")

    except Exception:
        log.exception("Scrape failed")
        db.rollback()
    finally:
        db.close()
        scraper.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UFC Stats scraper")
    parser.add_argument("--update", action="store_true", help="Only scrape new events since last in DB")
    parser.add_argument("--upcoming", action="store_true", help="Scrape upcoming events (next 4)")
    parser.add_argument("--recent", action="store_true", help="Fast post-event update (new completed events + refresh upcoming)")
    args = parser.parse_args()

    if args.recent:
        run_recent_update()
    elif args.upcoming:
        scrape_upcoming()
    else:
        run_scrape(mode="update" if args.update else "full")
