"""In-process stale-while-revalidate cache for the expensive read endpoints.

The database is remote, so every query is a ~240ms round trip, and a handful of
endpoints are nothing but round trips: /ufc/rankings (~6.5s, 2MB of precomputed
ledgers), /ufc/upcoming (~5s, five queries per card), a fight page (~1.8s). Their
inputs only change when a scheduled job writes, so recomputing them per request
is pure latency.

Semantics, per key:
  * miss            -> compute now; concurrent callers share the one computation.
  * fresh (< ttl)   -> served from memory.
  * stale (< max)   -> served from memory, and ONE background refresh is started,
                       so no user ever waits on a recompute after the first.
  * too stale       -> treated as a miss. Bounds how old an answer can be after a
                       quiet stretch (odds and cards should not be a day behind).

The scheduled jobs call `invalidate()` after they write and `warm()` to refill,
so a new card or ranking publish is visible without waiting out a TTL.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

log = logging.getLogger("response_cache")

#: Per-fight entries are ~100KB each; this caps the cache well under 50MB.
MAX_ENTRIES = 400


class _Entry:
    __slots__ = ("value", "stored_at", "refreshing")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.stored_at = time.monotonic()
        self.refreshing = False


_entries: OrderedDict[str, _Entry] = OrderedDict()
_inflight: dict[str, threading.Event] = {}
_lock = threading.Lock()
#: Bumped by invalidate(). A computation that began before an invalidation read the
#: tables as they were before the write, so its result is dropped rather than stored.
_generation = 0


def _store(key: str, value: Any, generation: int) -> None:
    with _lock:
        if generation != _generation:
            return
        _entries[key] = _Entry(value)
        _entries.move_to_end(key)
        while len(_entries) > MAX_ENTRIES:
            _entries.popitem(last=False)


def _compute(key: str, build: Callable[[], Any]) -> Any:
    """Run `build` once for `key`, however many threads ask at the same moment."""
    while True:
        with _lock:
            entry = _entries.get(key)
            if entry is not None and not entry.refreshing:
                # Another thread finished the computation we were waiting on.
                return entry.value
            waiter = _inflight.get(key)
            if waiter is None:
                done = threading.Event()
                _inflight[key] = done
                generation = _generation
                break
        waiter.wait()
        with _lock:
            entry = _entries.get(key)
        if entry is not None:
            return entry.value
        # The computation we waited on raised; take our own turn at it.

    try:
        value = build()
        _store(key, value, generation)
        return value
    finally:
        with _lock:
            _inflight.pop(key, None)
        done.set()


def _refresh_in_background(key: str, build: Callable[[], Any]) -> None:
    generation = _generation

    def run() -> None:
        try:
            _store(key, build(), generation)
        except Exception:
            # Keep serving the stale value; the next request past the TTL retries.
            log.exception("background refresh of %s failed", key)
            with _lock:
                entry = _entries.get(key)
                if entry is not None:
                    entry.refreshing = False

    threading.Thread(target=run, name=f"cache-refresh:{key}", daemon=True).start()


def cached(key: str, build: Callable[[], Any], ttl: float, max_stale: float = 3600) -> Any:
    """Return the cached value for `key`, computing it with `build()` as needed.

    `build` must open its own DB session (see `with_session`): a background
    refresh outlives the request that triggered it, and with it the request's.
    Exceptions from `build` propagate and are never cached, so a 404 stays a 404.
    """
    with _lock:
        entry = _entries.get(key)
        if entry is not None:
            age = time.monotonic() - entry.stored_at
            if age < ttl:
                _entries.move_to_end(key)
                return entry.value
            if age < max_stale:
                if not entry.refreshing:
                    entry.refreshing = True
                    _refresh_in_background(key, build)
                return entry.value
            del _entries[key]
    return _compute(key, build)


def with_session(fn: Callable[..., Any], *args: Any) -> Callable[[], Any]:
    """A `build` thunk that runs `fn(db, *args)` on a session of its own."""
    def build() -> Any:
        from app.database import SessionLocal

        db = SessionLocal()
        try:
            return fn(db, *args)
        finally:
            db.close()

    return build


def invalidate() -> None:
    """Drop everything. Called after a job writes the tables these views read."""
    global _generation
    with _lock:
        _generation += 1
        _entries.clear()


#: Registered by the routers: key -> build, for the views worth having hot before the
#: first visitor arrives. Kept here rather than imported so this module has no
#: dependency on the routers.
_warmers: dict[str, tuple[Callable[[], Any], float]] = {}


def register_warmer(key: str, build: Callable[[], Any], ttl: float) -> None:
    _warmers[key] = (build, ttl)


def warm() -> None:
    """Compute every registered view. Blocking — run it off the event loop."""
    for key, (build, ttl) in list(_warmers.items()):
        try:
            cached(key, build, ttl)
        except Exception:
            log.exception("warming %s failed", key)
