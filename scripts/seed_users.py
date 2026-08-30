"""Create the initial console accounts.

Usage:
    python scripts/seed_users.py                       # one account per role, random passwords
    python scripts/seed_users.py --user chief:supervisor:S0meL0ngPass
    python scripts/seed_users.py --list

The console ships with **no accounts**: an empty user table means nobody can
log in, which is the correct failure mode for a system with no owner yet.
This script is how the first ones exist.

Passwords are printed **once, to the terminal, and never stored in the clear**
— the database holds only an scrypt hash. If a password is lost the account is
re-seeded, not recovered. Generated passwords come from ``secrets``, not from
a word list.

For a demo, running this with no arguments gives one account per role, which
is exactly what showing the permission model to a judge needs.
"""

from __future__ import annotations

import argparse
import secrets
import string
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "contacts.db"

#: Demo accounts: one per role, so the matrix can be shown by logging in.
DEMO_USERS: tuple[tuple[str, str, str], ...] = (
    ("viewer", "viewer", "Ministry Observer"),
    ("operator", "operator", "Survey Technician"),
    ("analyst", "analyst", "Sonar Interpreter"),
    ("chief", "supervisor", "Survey Chief"),
    ("admin", "admin", "System Administrator"),
)

#: Long enough that the demo accounts are not a liability if the database
#: escapes, short enough to retype at a podium.
GENERATED_LEN = 16
ALPHABET = string.ascii_letters + string.digits


def generate_password(length: int = GENERATED_LEN) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def seed(db_path: Path, specs: list[tuple[str, str, str, str | None]]) -> None:
    """Create each (username, role, full_name, password) account."""
    from api.auth import ROLE_PERMISSIONS, Role, hash_password
    from api.db import ContactRepo

    repo = ContactRepo(db_path)
    print(f"database: {db_path}\n")
    rows = []
    for username, role_name, full_name, password in specs:
        try:
            role = Role(role_name)
        except ValueError:
            raise SystemExit(
                f"unknown role {role_name!r}; valid roles: "
                f"{', '.join(r.value for r in Role)}"
            ) from None
        pw = password or generate_password()
        repo.add_user(username, role.value, hash_password(pw), full_name)
        rows.append((username, role.value, pw, sorted(
            p.value for p in ROLE_PERMISSIONS[role]
        )))

    width = max(len(u) for u, *_ in rows)
    print(f"{'username'.ljust(width)}  {'role':<11}  password")
    print(f"{'-' * width}  {'-' * 11}  {'-' * GENERATED_LEN}")
    for username, role, pw, _perms in rows:
        print(f"{username.ljust(width)}  {role:<11}  {pw}")
    print("\npermissions granted:")
    for username, _role, _pw, perms in rows:
        print(f"  {username.ljust(width)}  {', '.join(perms)}")
    print(
        "\nThese passwords are shown once and are not recoverable — the database\n"
        "stores only an scrypt hash. Record them now or re-run to reset."
    )


def list_users(db_path: Path) -> None:
    from api.db import ContactRepo

    users = ContactRepo(db_path).list_users()
    if not users:
        print(f"no active users in {db_path} — run without --list to seed some")
        return
    width = max(len(u["username"]) for u in users)
    for u in users:
        print(f"{u['username'].ljust(width)}  {u['role']:<11}  {u['full_name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--user", action="append", default=[],
        metavar="NAME:ROLE[:PASSWORD]",
        help="seed one account; omit the password to have one generated",
    )
    parser.add_argument("--list", action="store_true", help="show active accounts")
    args = parser.parse_args()

    if args.list:
        list_users(args.db)
        return

    if args.user:
        specs = []
        for raw in args.user:
            parts = raw.split(":")
            if len(parts) not in (2, 3):
                raise SystemExit(f"malformed --user {raw!r}; expected NAME:ROLE[:PASSWORD]")
            username, role = parts[0], parts[1]
            password = parts[2] if len(parts) == 3 else None
            if password is not None and len(password) < 8:
                raise SystemExit("passwords must be at least 8 characters")
            specs.append((username, role, "", password))
    else:
        specs = [(u, r, n, None) for u, r, n in DEMO_USERS]

    seed(args.db, specs)


if __name__ == "__main__":
    main()
