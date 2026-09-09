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
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(username);
"""

#: Columns added after the first release. SQLite has no "ADD COLUMN IF NOT
#: EXISTS", so they are applied by inspecting pragma table_info at startup —
#: an existing survey database must not have to be thrown away to gain the
#: audit attribution that RBAC makes meaningful.
_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("reviews", "actor", "TEXT"),
    ("recovery_log", "actor", "TEXT"),
)


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
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Add post-release columns to tables that predate them.

        Called with the lock held. Idempotent: each column is added only when
        pragma table_info says it is absent, so repeated startups are a no-op
        and a database written by an older build keeps its rows.
        """
        for table, column, decl in _MIGRATIONS:
            cols = {
                r["name"]
                for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

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
        self,
        contact_id: str,
        status: ReviewStatus,
        notes: str | None = None,
        actor: str | None = None,
    ) -> Contact | None:
        """Record a verdict. *actor* is the username behind it.

        Attribution is what turns the review trail into defensible training
        data: "this label came from a certified analyst" is a claim only an
        attributed log can make. None is accepted so unauthenticated callers
        (tests, offline scripts) still work and are recorded as such.
        """
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
                "INSERT INTO reviews (contact_id, status, notes, at, actor) "
                "VALUES (?,?,?,?,?)",
                (
                    contact_id, status.value, notes,
                    datetime.now(tz=UTC).isoformat(timespec="seconds"), actor,
                ),
            )
            self._conn.commit()
        return updated

    def set_recovery(
        self, contact_id: str, status: RecoveryStatus, actor: str | None = None
    ) -> Contact | None:
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
                "INSERT INTO recovery_log (contact_id, status, at, actor) "
                "VALUES (?,?,?,?)",
                (
                    contact_id, status.value,
                    datetime.now(tz=UTC).isoformat(timespec="seconds"), actor,
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
                "SELECT contact_id, status, at, actor FROM recovery_log "
                "ORDER BY at, rowid"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------ users and sessions --
    # RBAC storage lives beside the survey data on purpose: the console is
    # offline-first and must not require a second service to answer "who is
    # this and what may they do".

    def add_user(
        self, username: str, role: str, password_hash: str, full_name: str = ""
    ) -> None:
        """Create or replace a user. The caller hashes the password."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO users "
                "(username, role, password_hash, full_name, created_at, active) "
                "VALUES (?,?,?,?,?,1)",
                (
                    username, role, password_hash, full_name,
                    datetime.now(tz=UTC).isoformat(timespec="seconds"),
                ),
            )
            self._conn.commit()

    def get_user(self, username: str) -> dict | None:
        """The full user row including the hash — for login only."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE username = ? AND active = 1", (username,)
            ).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list[dict]:
        """Every active user, without password hashes."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT username, role, full_name, created_at FROM users "
                "WHERE active = 1 ORDER BY username"
            ).fetchall()
        return [dict(r) for r in rows]

    def deactivate_user(self, username: str) -> bool:
        """Soft-delete: the user stops authenticating but their name stays
        resolvable in the review and recovery trails, which would otherwise
        develop holes wherever staff changed."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE users SET active = 0 WHERE username = ? AND active = 1",
                (username,),
            )
            self._conn.execute("DELETE FROM sessions WHERE username = ?", (username,))
            self._conn.commit()
        return cur.rowcount > 0

    def create_session(self, token_hash: str, username: str, expires_at: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(token_hash, username, created_at, expires_at) VALUES (?,?,?,?)",
                (
                    token_hash, username,
                    datetime.now(tz=UTC).isoformat(timespec="seconds"), expires_at,
                ),
            )
            self._conn.commit()

    def get_session(self, token_hash: str) -> dict | None:
        """Session joined to its user, or None if either is gone.

        The join means deactivating a user invalidates their live sessions
        without a sweep, and a session row orphaned by a deleted user can
        never authenticate.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT s.token_hash, s.username, s.expires_at, u.role, u.full_name "
                "FROM sessions s JOIN users u ON u.username = s.username "
                "WHERE s.token_hash = ? AND u.active = 1",
                (token_hash,),
            ).fetchone()
        return dict(row) if row else None

    def delete_session(self, token_hash: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM sessions WHERE token_hash = ?", (token_hash,)
            )
            self._conn.commit()

    def purge_expired_sessions(self, now_iso: str) -> int:
        """Drop sessions past their expiry. Called opportunistically on login
        so the table cannot grow without bound on a long-running console."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE expires_at <= ?", (now_iso,)
            )
            self._conn.commit()
        return cur.rowcount

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
