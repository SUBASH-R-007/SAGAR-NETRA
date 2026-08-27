"""Contact storage: SQLite by default (offline-first, zero setup).

Contacts are stored as their full pydantic JSON plus indexed scalar columns
for filtering; the review trail is append-only so confirmed/rejected labels
can later be exported as a retraining set. PostGIS can replace this behind
the same repository interface for shore-station deployments (docker-compose
ships one), but the demo path never requires a server.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from geoscribe.contact import Contact, RecoveryStatus, ReviewStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    id TEXT PRIMARY KEY,
    survey TEXT NOT NULL,
    cls TEXT NOT NULL,
    confidence REAL NOT NULL,
    severity REAL NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    review TEXT NOT NULL DEFAULT 'pending',
    detected_at TEXT,
    json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_contacts_survey ON contacts(survey);
CREATE INDEX IF NOT EXISTS idx_contacts_cls ON contacts(cls);
CREATE TABLE IF NOT EXISTS reviews (
    contact_id TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT,
    at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS surveys (
    name TEXT PRIMARY KEY,
    source_path TEXT,
    processed_at TEXT,
    n_pings INTEGER,
    n_contacts INTEGER,
    outputs_dir TEXT
);
CREATE TABLE IF NOT EXISTS recovery_log (
    contact_id TEXT NOT NULL,
    status TEXT NOT NULL,
    at TEXT NOT NULL
);
"""


class ContactRepo:
    """Thread-safe SQLite repository (one connection, serialized writes)."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ writes --

    def add_contacts(self, contacts: list[Contact]) -> None:
        rows = [
            (
                c.id, c.survey, c.cls, c.confidence, c.severity, c.lat, c.lon,
                c.review.value, c.detected_at, c.model_dump_json(),
            )
            for c in contacts
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO contacts VALUES (?,?,?,?,?,?,?,?,?,?)", rows
            )
            self._conn.commit()

    def upsert_survey(
        self,
        name: str,
        source_path: str,
        n_pings: int,
        n_contacts: int,
        outputs_dir: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO surveys VALUES (?,?,?,?,?,?)",
                (
                    name, source_path, datetime.now(tz=UTC).isoformat(timespec="seconds"),
                    n_pings, n_contacts, outputs_dir,
                ),
            )
            self._conn.commit()

    def set_review(
        self, contact_id: str, status: ReviewStatus, notes: str | None = None
    ) -> Contact | None:
        contact = self.get(contact_id)
        if contact is None:
            return None
        updated = contact.model_copy(update={"review": status, "notes": notes})
        with self._lock:
            self._conn.execute(
                "UPDATE contacts SET review = ?, json = ? WHERE id = ?",
                (status.value, updated.model_dump_json(), contact_id),
            )
            self._conn.execute(
                "INSERT INTO reviews VALUES (?,?,?,?)",
                (
                    contact_id, status.value, notes,
                    datetime.now(tz=UTC).isoformat(timespec="seconds"),
                ),
            )
            self._conn.commit()
        return updated

    def set_recovery(self, contact_id: str, status: RecoveryStatus) -> Contact | None:
        """Advance the physical recovery workflow (flagged -> assigned -> retrieved).

        Mirrors :meth:`set_review`: the contact JSON is updated in place and an
        append-only ``recovery_log`` row records who-retrieved-what-when for the
        operations audit trail. The table is created by ``_SCHEMA`` with IF NOT
        EXISTS, so existing databases pick it up with no migration.
        """
        contact = self.get(contact_id)
        if contact is None:
            return None
        updated = contact.model_copy(update={"recovery": status})
        with self._lock:
            self._conn.execute(
                "UPDATE contacts SET json = ? WHERE id = ?",
                (updated.model_dump_json(), contact_id),
            )
            self._conn.execute(
                "INSERT INTO recovery_log VALUES (?,?,?)",
                (
                    contact_id, status.value,
                    datetime.now(tz=UTC).isoformat(timespec="seconds"),
                ),
            )
            self._conn.commit()
        return updated

    def delete_survey(self, name: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM contacts WHERE survey = ?", (name,))
            self._conn.execute("DELETE FROM surveys WHERE name = ?", (name,))
            self._conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------- reads --

    def get(self, contact_id: str) -> Contact | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT json FROM contacts WHERE id = ?", (contact_id,)
            ).fetchone()
        return Contact.model_validate_json(row["json"]) if row else None

    def query(
        self,
        survey: str | None = None,
        cls: str | None = None,
        min_conf: float | None = None,
        min_sev: float | None = None,
        review: str | None = None,
        limit: int = 500,
    ) -> list[Contact]:
        clauses, params = [], []
        if survey:
            clauses.append("survey = ?")
            params.append(survey)
        if cls:
            clauses.append("cls = ?")
            params.append(cls)
        if min_conf is not None:
            clauses.append("confidence >= ?")
            params.append(min_conf)
        if min_sev is not None:
            clauses.append("severity >= ?")
            params.append(min_sev)
        if review:
            clauses.append("review = ?")
            params.append(review)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT json FROM contacts {where} ORDER BY severity DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [Contact.model_validate_json(r["json"]) for r in rows]

    def surveys(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM surveys ORDER BY processed_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def review_log(self) -> list[dict]:
        """Append-only review trail — the future retraining label export."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM reviews ORDER BY at").fetchall()
        return [dict(r) for r in rows]

    def recovery_log(self) -> list[dict]:
        """Append-only recovery audit trail, oldest first (rowid breaks
        same-second timestamp ties so the workflow order is never ambiguous)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT contact_id, status, at FROM recovery_log ORDER BY at, rowid"
            ).fetchall()
        return [dict(r) for r in rows]

    def run_sql(self, sql: str, params: tuple = ()) -> list[dict]:
        """Read-only query hook for the copilot (SELECT-only, enforced)."""
        lowered = sql.strip().lower()
        if not lowered.startswith("select"):
            raise ValueError("copilot queries must be SELECT statements")
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {k: _maybe_json(v) if k == "json" else v for k, v in dict(r).items()} for r in rows
        ]


def _maybe_json(value):
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value
