"""Role-based access control: roles, permissions, passwords, sessions.

The console drives a real operational workflow — a vessel technician uploads a
survey, a trained interpreter judges whether each contact is real, a survey
chief commits a boat to going and retrieving it. Those are different people
with different training, and the permission model follows that split rather
than a generic admin/user tier.

Five roles, each a superset of the one before except where the workflow says
otherwise:

* ``viewer``     — read everything, change nothing. Ministry, partners, press.
* ``operator``   — read + **upload**. The technician on the vessel: they run
  the sonar and need to see whether the line worked, but judging a detection
  is an interpretation task they are not trained for.
* ``analyst``    — read + **review**. Confirms or rejects contacts; every
  verdict becomes a training label, which is why it is a distinct permission.
* ``supervisor`` — analyst + operator + **recovery** transitions and
  **survey deletion**. Committing a retrieval asset is an operational
  decision; deleting a survey destroys review verdicts, i.e. training data.
* ``admin``      — everything, plus user management.

Two deliberate choices worth knowing:

**Enforcement lives on the API, not the UI.** Hiding a button is not access
control — ``curl`` ignores it. Every guard here is a FastAPI dependency, and
the console hides what a user cannot do purely as a courtesy.

**Passwords use `hashlib.scrypt` from the standard library.** bcrypt/argon2
would be conventional, but this system's whole posture is offline-first with
no surprise dependencies, and scrypt is a memory-hard KDF designed for exactly
this. Parameters are stored beside each hash so they can be raised later
without invalidating existing users.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

#: scrypt work factors. n=2**14 keeps a single verification near ~50 ms on a
#: laptop and well under a second on a Pi — deliberate: login is rare, and the
#: cost is what makes an offline copy of the user table expensive to attack.
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SALT_BYTES = 16

#: Sessions expire on their own so an unattended console on a vessel does not
#: stay authenticated indefinitely.
SESSION_TTL = timedelta(hours=12)
SESSION_COOKIE = "sagar_session"


class Role(StrEnum):
    viewer = "viewer"
    operator = "operator"
    analyst = "analyst"
    supervisor = "supervisor"
    admin = "admin"


class Permission(StrEnum):
    """What a request is allowed to do, named for the operational act."""

    read = "read"  # contacts, imagery, reports, copilot, physics lab
    upload = "upload"  # ingest a survey, pick a mission profile
    review = "review"  # confirm / reject a contact (creates training labels)
    recover = "recover"  # flagged -> assigned -> retrieved
    delete_survey = "delete_survey"  # destroys contacts AND their verdicts
    manage_users = "manage_users"


#: The matrix. Written out per role rather than by inheritance so that reading
#: one line tells you everything that role can do — an inheritance chain hides
#: exactly the question an auditor asks.
ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.viewer: frozenset({Permission.read}),
    Role.operator: frozenset({Permission.read, Permission.upload}),
    Role.analyst: frozenset({Permission.read, Permission.review}),
    Role.supervisor: frozenset({
        Permission.read, Permission.upload, Permission.review,
        Permission.recover, Permission.delete_survey,
    }),
    Role.admin: frozenset(Permission),
}


@dataclass(frozen=True)
class User:
    """An authenticated principal. Never carries the password hash."""

    username: str
    role: Role
    full_name: str = ""

    @property
    def permissions(self) -> frozenset[Permission]:
        return ROLE_PERMISSIONS[self.role]

    def can(self, permission: Permission) -> bool:
        return permission in self.permissions


# ------------------------------------------------------------- passwords --


def hash_password(password: str) -> str:
    """``scrypt$n$r$p$salt$hash`` — parameters travel with the hash.

    Storing the work factors inline means they can be raised later without a
    flag day: an old hash still verifies under its own parameters, and can be
    re-hashed on next successful login.
    """
    if not password:
        raise ValueError("password must not be empty")
    salt = secrets.token_bytes(SALT_BYTES)
    dk = hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=SCRYPT_DKLEN,
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${dk.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time verification against a stored hash.

    Returns False for malformed input rather than raising: a corrupted row in
    the user table must fail the login, not the whole request handler.
    """
    try:
        scheme, n, r, p, salt_hex, hash_hex = encoded.split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p), dklen=len(hash_hex) // 2,
        )
    except (ValueError, TypeError, MemoryError):
        return False
    # compare_digest: a byte-by-byte comparison would leak the prefix length
    # through timing.
    return hmac.compare_digest(dk.hex(), hash_hex)


#: Verified against when the username does not exist, so a missing user and a
#: wrong password cost the same work. Without it, login latency alone reveals
#: which usernames are real. Generated once at import from an unguessable
#: value: it must never match a password anyone could supply.
DUMMY_HASH = hash_password(secrets.token_urlsafe(32))


# -------------------------------------------------------------- sessions --


def new_session_token() -> str:
    """A fresh opaque session token. 32 bytes of urandom, URL-safe."""
    return secrets.token_urlsafe(32)


def token_fingerprint(token: str) -> str:
    """What gets stored server-side.

    The database holds only a SHA-256 of the token, so a leaked database
    cannot be replayed as a live session — the same reason password hashes are
    not stored in the clear. Plain SHA-256 (not a KDF) is right here: the token
    is already 256 bits of entropy, so there is nothing to brute-force.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def session_expiry(now: datetime | None = None, ttl: timedelta = SESSION_TTL) -> str:
    return ((now or datetime.now(tz=UTC)) + ttl).isoformat()


def is_expired(expires_at: str, now: datetime | None = None) -> bool:
    """True when a stored expiry has passed, or cannot be parsed.

    An unparseable expiry is treated as expired: the safe direction for a
    field that gates access.
    """
    try:
        return datetime.fromisoformat(expires_at) <= (now or datetime.now(tz=UTC))
    except (ValueError, TypeError):
        return True
