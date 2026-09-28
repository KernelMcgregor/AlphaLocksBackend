"""Persistent audit log of every action that writes to production.

Every admin button, scheduled job, GitHub Actions workflow and manual record edit opens
a row in admin_action_runs and closes it with a status. Steps inside a pipeline become
child rows of the run that is active on the current thread, so a nightly job reads as
one entry with its steps underneath rather than a flat list of unrelated labels.

Auditing must never be the reason a job fails: every write here uses its own session
and swallows its own errors after logging them.

CLI, for GitHub Actions, which reaches the database directly rather than through the API:

    RUN_ID=$(python -m app.services.audit start --action "Post-Event Update" --source github)
    python -m app.services.audit finish --id "$RUN_ID" --status success
"""
import argparse
import json
import logging
import threading
from contextlib import contextmanager
from datetime import datetime

from app.database import SessionLocal
from app.models.shared import AdminActionRun

log = logging.getLogger(__name__)

#: The run a job is executing under, per thread. Background tasks and scheduler jobs each
#: run on a worker thread, so this is what lets record_step() find its parent without
#: every service function taking a run id argument.
_local = threading.local()


def _dump(value) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, default=str)
    except Exception:
        return json.dumps(str(value))


def _stack() -> list[dict]:
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


def current_run() -> dict | None:
    """{"id", "action", "source", "actor", "errors"} for the innermost active run, or None."""
    stack = _stack()
    return stack[-1] if stack else None


def start_run(
    action: str,
    source: str,
    actor: str | None = None,
    params: dict | None = None,
    target: str | None = None,
    parent_id: int | None = None,
) -> int | None:
    """Insert a "running" row and return its id (None if the audit write itself failed)."""
    db = SessionLocal()
    try:
        row = AdminActionRun(
            parent_id=parent_id, action=action[:120], source=source, actor=(actor or None) and actor[:120],
            status="running", target=target, params=_dump(params), started_at=datetime.utcnow(),
        )
        db.add(row)
        db.commit()
        return row.id
    except Exception:
        log.exception(f"audit: could not open run for {action}")
        db.rollback()
        return None
    finally:
        db.close()


def finish_run(run_id: int | None, status: str, error: str | None = None, summary=None) -> None:
    if run_id is None:
        return
    db = SessionLocal()
    try:
        row = db.get(AdminActionRun, run_id)
        if row is None:
            return
        row.status = status
        row.error = error
        if summary is not None:
            row.summary = _dump(summary)
        row.finished_at = datetime.utcnow()
        db.commit()
    except Exception:
        log.exception(f"audit: could not close run {run_id}")
        db.rollback()
    finally:
        db.close()


def record_step(label: str, status: str, error: str | None = None, started: datetime | None = None) -> None:
    """Record one finished step as a child of the active run.

    A step with the same label as its parent is the job reporting on itself (the scheduled
    wrappers call record_run with their own label), so it is folded into the parent
    instead of duplicated underneath it.
    """
    parent = current_run()
    if parent and error:
        parent["errors"].append(f"{label}: {error}")
    if parent and label == parent["action"]:
        return

    db = SessionLocal()
    try:
        now = datetime.utcnow()
        db.add(AdminActionRun(
            parent_id=parent["id"] if parent else None,
            action=label[:120],
            source=parent["source"] if parent else "scheduled",
            actor=parent["actor"] if parent else "scheduler",
            status="done" if status == "done" else "error",
            error=error,
            started_at=started or now,
            finished_at=now,
        ))
        db.commit()
    except Exception:
        log.exception(f"audit: could not record step {label}")
        db.rollback()
    finally:
        db.close()


@contextmanager
def step(label: str):
    """Run one pipeline step, visible as a "running" child row while it runs.

    record_step() only writes once a step has finished, so a long step (winner
    predictions over every fight) left the dashboard showing the run as running with no
    hint of where. Outside any tracked run this still records the step, at the end.
    """
    parent = current_run()
    if parent is None or label == parent["action"]:
        started = datetime.utcnow()
        try:
            yield
        except Exception as e:
            record_step(label, "error", str(e), started)
            raise
        record_step(label, "done", None, started)
        return

    run_id = start_run(label, parent["source"], parent["actor"], parent_id=parent["id"])
    try:
        yield
    except Exception as e:
        parent["errors"].append(f"{label}: {e}")
        finish_run(run_id, "error", str(e))
        raise
    finish_run(run_id, "done")


@contextmanager
def track(
    action: str,
    source: str,
    actor: str | None = None,
    params: dict | None = None,
    target: str | None = None,
    run_id: int | None = None,
):
    """Run a block as an audited action. Pass run_id to adopt a row opened earlier (the
    admin API opens it at request time so the caller gets an id to poll).

    Yields a dict whose "summary" key, if set, is stored on the row. Finishes as
    "error" if the block raised (the exception propagates), "partial" if it returned
    but a step recorded an error, otherwise "done".
    """
    if run_id is None:
        parent = current_run()
        run_id = start_run(action, source, actor, params, target, parent["id"] if parent else None)
    frame = {"id": run_id, "action": action, "source": source, "actor": actor, "errors": [], "summary": None}
    stack = _stack()
    stack.append(frame)
    try:
        yield frame
    except Exception as e:
        finish_run(run_id, "error", f"{type(e).__name__}: {e}", frame["summary"])
        raise
    else:
        if frame["errors"]:
            n = len(frame["errors"])
            finish_run(run_id, "partial", f"{n} step{'s' if n != 1 else ''} failed:\n" + "\n".join(frame["errors"]),
                       frame["summary"])
        else:
            finish_run(run_id, "done", None, frame["summary"])
    finally:
        # Worker threads are pooled, so a frame left behind would adopt the next job's steps.
        stack.pop()


def record_event(
    action: str,
    source: str,
    actor: str | None,
    target: str | None = None,
    params: dict | None = None,
    summary=None,
) -> int | None:
    """An instantaneous, already-completed action, such as a manual record edit."""
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        row = AdminActionRun(
            action=action[:120], source=source, actor=actor, status="done", target=target,
            params=_dump(params), summary=_dump(summary), started_at=now, finished_at=now,
        )
        db.add(row)
        db.commit()
        return row.id
    except Exception:
        log.exception(f"audit: could not record {action}")
        db.rollback()
        return None
    finally:
        db.close()


def mark_interrupted() -> None:
    """Close out rows left "running" by a process that died (deploy, crash, OOM).

    Called at startup, before the scheduler starts, so it cannot catch a live job. Only
    in-app runs are touched: a GitHub Actions run is a separate process that may well be
    running right now.
    """
    db = SessionLocal()
    try:
        n = (
            db.query(AdminActionRun)
            .filter(AdminActionRun.status == "running", AdminActionRun.source.in_(("manual", "scheduled")))
            .update({"status": "interrupted", "finished_at": datetime.utcnow(),
                     "error": "Server restarted before this finished"}, synchronize_session=False)
        )
        db.commit()
        if n:
            log.warning(f"audit: marked {n} unfinished run(s) as interrupted")
    except Exception:
        log.exception("audit: could not mark interrupted runs")
        db.rollback()
    finally:
        db.close()


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Open or close an audit-log row (used by GitHub Actions)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--action", required=True)
    s.add_argument("--source", default="github")
    s.add_argument("--actor")
    s.add_argument("--url", help="Link back to the workflow run")
    f = sub.add_parser("finish")
    f.add_argument("--id", default="")
    # GitHub's job.status vocabulary: success | failure | cancelled
    f.add_argument("--status", required=True)
    f.add_argument("--summary")
    args = parser.parse_args()

    if args.cmd == "start":
        run_id = start_run(args.action, args.source, args.actor, {"url": args.url} if args.url else None)
        # Printed for $(...) capture. Empty on failure, so finish becomes a no-op rather
        # than the audit log failing the workflow.
        print(run_id or "")
    else:
        if not args.id.strip():
            return
        status = {"success": "done", "failure": "error", "cancelled": "interrupted"}.get(args.status, args.status)
        error = None if status == "done" else f"Workflow {args.status} — see the run's logs on GitHub"
        finish_run(int(args.id), status, error, {"note": args.summary} if args.summary else None)


if __name__ == "__main__":
    _cli()
