"""DeepSeek calls for the newsroom, each metered against the monthly budget."""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import threading

from app.config import settings

log = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    pass


class Meter:
    """Tokens and dollars for one run. `persist` writes each call to ufc_newsroom_spend
    so the cap holds across runs; a dry run only counts."""

    def __init__(self, persist: bool, month_spent: float = 0.0):
        self.persist = persist
        self.month_spent = month_spent
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    @property
    def run_usd(self) -> float:
        return sum(c["usd"] for c in self.calls)

    def check(self, reserve: float = 0.0):
        cap = settings.NEWSROOM_MONTHLY_BUDGET_USD * 0.9
        if self.month_spent + self.run_usd + reserve > cap:
            raise BudgetExceeded(
                f"Newsroom budget: ${self.month_spent + self.run_usd:.2f} of ${cap:.2f} used this month")

    def record(self, stage: str, fight_id: int | None, usage) -> float:
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        usd = (pt * settings.NEWSROOM_PRICE_IN_PER_M + ct * settings.NEWSROOM_PRICE_OUT_PER_M) / 1e6
        row = {"stage": stage, "fight_id": fight_id, "prompt_tokens": pt, "completion_tokens": ct, "usd": usd}
        with self._lock:
            self.calls.append(row)
        if self.persist:
            from app.database import SessionLocal
            from app.models.ufc import UFCNewsroomSpend
            db = SessionLocal()
            try:
                db.add(UFCNewsroomSpend(model=settings.PREVIEW_MODEL, **row))
                db.commit()
            finally:
                db.close()
        return usd


def month_spent(db) -> float:
    from sqlalchemy import func
    from app.models.ufc import UFCNewsroomSpend
    start = dt.datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return float(db.query(func.coalesce(func.sum(UFCNewsroomSpend.usd), 0.0))
                 .filter(UFCNewsroomSpend.created_at >= start).scalar() or 0.0)


_client = None


def client():
    global _client
    if _client is None:
        from app.services.ufc.preview_service import _preview_client
        _client = _preview_client()
    return _client


def chat(meter: Meter, stage: str, fight_id: int | None, system: str, user: str,
         json_mode: bool = False, max_tokens: int = 4096) -> str:
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    meter.check()
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client().chat.completions.create(
        model=settings.PREVIEW_MODEL, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
    meter.record(stage, fight_id, resp.usage)
    return (resp.choices[0].message.content or "").strip()


def chat_json(meter: Meter, stage: str, fight_id: int | None, system: str, user: str,
              max_tokens: int = 4096) -> dict:
    out = chat(meter, stage, fight_id, system, user, json_mode=True, max_tokens=max_tokens)
    out = re.sub(r"^```(?:json)?\s*|\s*```$", "", out)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        log.warning(f"{stage}: reply was not JSON")
        return {}
