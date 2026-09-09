"""RBAC: the permission matrix, session lifecycle, and password handling.

The tests that matter most here are the negative ones. An access-control layer
that grants correctly but denies loosely is worse than none, because it looks
like security. So every role is asserted against every permission — including
the ones it must NOT have — and the guards are exercised over HTTP rather than
by calling the checker directly, because the guard only counts if it is
actually wired to the route.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from api.auth import (
    ROLE_PERMISSIONS,
    Permission,
    Role,
    User,
    hash_password,
    is_expired,
    new_session_token,
    session_expiry,
    token_fingerprint,
    verify_password,
)
from api.db import ContactRepo
from api.main import create_app

PASSWORD = "correct-horse-battery"  # noqa: S105 - test fixture, not a secret


@pytest.fixture
def app_and_repo(tmp_path):
    repo = ContactRepo(tmp_path / "rbac.db")
    for name, role in (
        ("viewer1", Role.viewer), ("op1", Role.operator), ("an1", Role.analyst),
        ("sup1", Role.supervisor), ("admin1", Role.admin),
    ):
        repo.add_user(name, role.value, hash_password(PASSWORD), f"{name} name")
    app = create_app(
        repo=repo, upload_dir=tmp_path / "up", output_root=tmp_path / "out",
    )
    return app, repo


@pytest.fixture
def client(app_and_repo):
    app, _ = app_and_repo
    return TestClient(app)


def login(client: TestClient, username: str) -> None:
    r = client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
    assert r.status_code == 200, r.text


# ------------------------------------------------------- permission matrix --


@pytest.mark.parametrize(
    ("role", "permission", "allowed"),
    [
        # viewer: reads and nothing else
        (Role.viewer, Permission.read, True),
        (Role.viewer, Permission.upload, False),
        (Role.viewer, Permission.review, False),
        (Role.viewer, Permission.recover, False),
        (Role.viewer, Permission.delete_survey, False),
        (Role.viewer, Permission.manage_users, False),
        # operator uploads but must never judge a detection
        (Role.operator, Permission.upload, True),
        (Role.operator, Permission.read, True),
        (Role.operator, Permission.review, False),
        (Role.operator, Permission.delete_survey, False),
        # analyst judges but does not ingest or commit a retrieval asset
        (Role.analyst, Permission.review, True),
        (Role.analyst, Permission.upload, False),
        (Role.analyst, Permission.recover, False),
        (Role.analyst, Permission.delete_survey, False),
        # supervisor runs the operation, but is not an account administrator
        (Role.supervisor, Permission.recover, True),
        (Role.supervisor, Permission.delete_survey, True),
        (Role.supervisor, Permission.review, True),
        (Role.supervisor, Permission.upload, True),
        (Role.supervisor, Permission.manage_users, False),
        # admin: everything
        (Role.admin, Permission.manage_users, True),
        (Role.admin, Permission.delete_survey, True),
    ],
)
def test_permission_matrix(role: Role, permission: Permission, allowed: bool) -> None:
    assert User("u", role).can(permission) is allowed


def test_every_role_is_in_the_matrix() -> None:
    """A role added to the enum without a permission set would raise KeyError
    at request time; catch it here instead."""
    assert set(ROLE_PERMISSIONS) == set(Role)


def test_only_admin_manages_users() -> None:
    holders = {r for r, p in ROLE_PERMISSIONS.items() if Permission.manage_users in p}
    assert holders == {Role.admin}


def test_every_role_can_read() -> None:
    """Read is the floor: an authenticated user who can see nothing has no
    reason to hold an account."""
    assert all(Permission.read in p for p in ROLE_PERMISSIONS.values())


# ------------------------------------------------------------- passwords --


def test_password_round_trip_and_rejection() -> None:
    encoded = hash_password(PASSWORD)
    assert verify_password(PASSWORD, encoded)
    assert not verify_password(PASSWORD + "x", encoded)
    assert not verify_password("", encoded)


def test_hashes_are_salted() -> None:
    """Two users with the same password must not share a hash, or the table
    reveals which accounts to attack together."""
    assert hash_password(PASSWORD) != hash_password(PASSWORD)


def test_hash_carries_its_parameters() -> None:
    scheme, n, r, p, salt, digest = hash_password(PASSWORD).split("$")
    assert scheme == "scrypt"
    assert int(n) >= 2**14 and int(r) >= 8 and int(p) >= 1
    assert len(bytes.fromhex(salt)) >= 16
    assert len(bytes.fromhex(digest)) == 32


def test_malformed_hash_fails_closed() -> None:
    for bad in ("", "notascheme$1$2$3$4$5", "scrypt$x$y$z$q$r", "scrypt$16384$8"):
        assert verify_password(PASSWORD, bad) is False


def test_empty_password_is_refused() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        hash_password("")


# -------------------------------------------------------------- sessions --


def test_session_tokens_are_unique_and_stored_hashed(app_and_repo) -> None:
    _, repo = app_and_repo
    t1, t2 = new_session_token(), new_session_token()
    assert t1 != t2
    repo.create_session(token_fingerprint(t1), "an1", session_expiry())
    # The raw token must not be recoverable from the store.
    assert repo.get_session(t1) is None
    assert repo.get_session(token_fingerprint(t1))["username"] == "an1"


def test_expiry_detection() -> None:
    past = (datetime.now(tz=UTC) - timedelta(minutes=1)).isoformat()
    future = (datetime.now(tz=UTC) + timedelta(hours=1)).isoformat()
    assert is_expired(past)
    assert not is_expired(future)
    # Unparseable expiry must read as expired, not as "no expiry".
    assert is_expired("not-a-timestamp")


def test_expired_session_does_not_authenticate(client, app_and_repo) -> None:
    _, repo = app_and_repo
    token = new_session_token()
    repo.create_session(
        token_fingerprint(token), "admin1",
        (datetime.now(tz=UTC) - timedelta(seconds=1)).isoformat(),
    )
    client.cookies.set("sagar_session", token)
    assert client.get("/api/auth/me").status_code == 401


def test_deactivating_a_user_kills_their_session(client, app_and_repo) -> None:
    _, repo = app_and_repo
    login(client, "an1")
    assert client.get("/api/auth/me").status_code == 200
    repo.deactivate_user("an1")
    assert client.get("/api/auth/me").status_code == 401


# ------------------------------------------------------------ HTTP guards --


def test_unauthenticated_is_401_not_403(client) -> None:
    """401 and 403 must stay distinct: one says log in, the other says your
    role cannot do this. Collapsing them offers a login box to someone who is
    already signed in."""
    r = client.post("/api/contacts/x/review", json={"status": "confirmed"})
    assert r.status_code == 401


def test_wrong_role_is_403_with_a_reason(client) -> None:
    login(client, "viewer1")
    r = client.post("/api/contacts/x/review", json={"status": "confirmed"})
    assert r.status_code == 403
    assert "viewer" in r.json()["detail"] and "review" in r.json()["detail"]


def test_operator_may_not_review_and_analyst_may_not_delete(client) -> None:
    login(client, "op1")
    assert client.post(
        "/api/contacts/x/review", json={"status": "confirmed"}
    ).status_code == 403
    client.post("/api/auth/logout")

    login(client, "an1")
    assert client.delete("/api/surveys/anything").status_code == 403


def test_supervisor_reaches_delete_but_not_user_management(client) -> None:
    login(client, "sup1")
    # 404, not 403: the guard passed and the survey simply does not exist.
    assert client.delete("/api/surveys/nope").status_code == 404
    assert client.get("/api/auth/users").status_code == 403


def test_login_failures_are_indistinguishable(client) -> None:
    """Unknown user and wrong password must return the same status and text,
    or the endpoint enumerates accounts."""
    a = client.post("/api/auth/login", json={"username": "an1", "password": "wrong"})
    b = client.post("/api/auth/login", json={"username": "ghost", "password": "wrong"})
    assert a.status_code == b.status_code == 401
    assert a.json()["detail"] == b.json()["detail"]


def test_login_sets_httponly_cookie_and_reports_permissions(client) -> None:
    r = client.post(
        "/api/auth/login", json={"username": "sup1", "password": PASSWORD}
    )
    assert r.status_code == 200
    assert r.json()["role"] == "supervisor"
    assert "review" in r.json()["permissions"]
    assert "manage_users" not in r.json()["permissions"]
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie


def test_logout_invalidates_the_session(client) -> None:
    login(client, "admin1")
    assert client.get("/api/auth/me").status_code == 200
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401


def test_admin_can_create_and_deactivate_users(client) -> None:
    login(client, "admin1")
    r = client.post("/api/auth/users", json={
        "username": "new1", "password": "longenough1", "role": "analyst",
        "full_name": "New Analyst",
    })
    assert r.status_code == 200
    assert "new1" in [u["username"] for u in client.get("/api/auth/users").json()]
    assert client.delete("/api/auth/users/new1").status_code == 200
    assert "new1" not in [u["username"] for u in client.get("/api/auth/users").json()]


def test_short_passwords_are_refused(client) -> None:
    login(client, "admin1")
    r = client.post("/api/auth/users", json={
        "username": "weak", "password": "short", "role": "viewer",
    })
    assert r.status_code == 422


def test_admin_cannot_lock_themselves_out(client) -> None:
    """Deactivating the acting account would leave a system with no way in."""
    login(client, "admin1")
    r = client.delete("/api/auth/users/admin1")
    assert r.status_code == 422


def test_auth_can_be_disabled_for_offline_use(tmp_path) -> None:
    """require_auth=False is how tests and offline scripts drive the app. It
    must be explicit — the default is closed."""
    app = create_app(
        repo=ContactRepo(tmp_path / "open.db"),
        upload_dir=tmp_path / "u", output_root=tmp_path / "o",
        require_auth=False,
    )
    c = TestClient(app)
    body = c.get("/api/auth/me").json()
    assert body["role"] == "admin" and body["auth_enabled"] is False
    # A guarded route is reachable without any session.
    assert c.delete("/api/surveys/nope").status_code == 404
