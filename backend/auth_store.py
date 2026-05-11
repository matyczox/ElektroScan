from __future__ import annotations

import base64
import hashlib
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "elektroscan.db"
PASSWORD_ITERATIONS = 310_000


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat()


def init_database() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS auth_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                created_at_utc TEXT NOT NULL,
                last_seen_at_utc TEXT NOT NULL,
                expires_at_utc TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS auth_tokens (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                purpose TEXT NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                created_at_utc TEXT NOT NULL,
                expires_at_utc TEXT NOT NULL,
                consumed_at_utc TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                owner_user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                archived_at_utc TEXT,
                FOREIGN KEY(owner_user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS project_upload_sessions (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                source_pdf TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                last_accessed_at_utc TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS analysis_runs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                source_pdf TEXT NOT NULL,
                snapshot_path TEXT NOT NULL DEFAULT '',
                generated_at_utc TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY(session_id) REFERENCES project_upload_sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_auth_sessions_token_hash
                ON auth_sessions(token_hash);
            CREATE INDEX IF NOT EXISTS idx_auth_tokens_token_hash
                ON auth_tokens(token_hash);
            CREATE INDEX IF NOT EXISTS idx_auth_tokens_user_purpose
                ON auth_tokens(user_id, purpose, consumed_at_utc, expires_at_utc);
            CREATE INDEX IF NOT EXISTS idx_projects_owner
                ON projects(owner_user_id, archived_at_utc, updated_at_utc);
            CREATE INDEX IF NOT EXISTS idx_upload_sessions_project
                ON project_upload_sessions(project_id, created_at_utc);
            CREATE INDEX IF NOT EXISTS idx_analysis_runs_project
                ON analysis_runs(project_id, generated_at_utc);
            """
        )


def create_user(*, email: str, password: str, name: str | None = None) -> dict[str, Any]:
    clean_email = normalize_email(email)
    if not clean_email:
        raise ValueError("Podaj poprawny adres e-mail.")
    if len(password) < 8:
        raise ValueError("Hasło musi mieć co najmniej 8 znaków.")

    user_id = str(uuid.uuid4())
    display_name = (name or clean_email.split("@")[0]).strip() or clean_email
    now = utc_iso()
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO users (
                    id, email, name, password_hash, created_at_utc, updated_at_utc
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, clean_email, display_name, hash_password(password), now, now),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError("Konto z takim adresem e-mail już istnieje.") from exc

    user = get_user_by_id(user_id)
    if user is None:
        raise RuntimeError("Nie udało się utworzyć użytkownika.")
    return user


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    clean_email = normalize_email(email)
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (clean_email,),
        ).fetchone()
    if row is None:
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    return public_user(dict(row))


def create_auth_session(user_id: str, *, ttl_days: int = 30) -> tuple[str, dict[str, Any]]:
    token = secrets.token_urlsafe(32)
    token_hash = hash_session_token(token)
    session_id = str(uuid.uuid4())
    now = utc_now()
    expires_at = now + timedelta(days=ttl_days)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO auth_sessions (
                id, user_id, token_hash, created_at_utc, last_seen_at_utc, expires_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, user_id, token_hash, utc_iso(now), utc_iso(now), utc_iso(expires_at)),
        )
    session = {
        "id": session_id,
        "userId": user_id,
        "expiresAtUtc": utc_iso(expires_at),
    }
    return token, session


def list_auth_sessions_for_user(
    user_id: str,
    *,
    current_token: str | None = None,
) -> list[dict[str, Any]]:
    now = utc_iso()
    current_token_hash = hash_session_token(current_token) if current_token else None
    with _connect() as conn:
        conn.execute("DELETE FROM auth_sessions WHERE expires_at_utc <= ?", (now,))
        rows = conn.execute(
            """
            SELECT id, user_id, token_hash, created_at_utc, last_seen_at_utc, expires_at_utc
            FROM auth_sessions
            WHERE user_id = ? AND expires_at_utc > ?
            ORDER BY last_seen_at_utc DESC, created_at_utc DESC
            """,
            (user_id, now),
        ).fetchall()
    return [auth_session_payload(dict(row), current_token_hash) for row in rows]


def get_user_by_session_token(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    token_hash = hash_session_token(token)
    now = utc_iso()
    with _connect() as conn:
        conn.execute("DELETE FROM auth_sessions WHERE expires_at_utc <= ?", (now,))
        row = conn.execute(
            """
            SELECT users.*
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token_hash = ?
              AND auth_sessions.expires_at_utc > ?
            """,
            (token_hash, now),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE auth_sessions SET last_seen_at_utc = ? WHERE token_hash = ?",
            (now, token_hash),
        )
    return public_user(dict(row))


def delete_auth_session(token: str | None) -> None:
    if not token:
        return
    with _connect() as conn:
        conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (hash_session_token(token),))


def delete_auth_session_by_id(user_id: str, session_id: str) -> bool:
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM auth_sessions WHERE user_id = ? AND id = ?",
            (user_id, session_id),
        )
    return cursor.rowcount > 0


def delete_auth_sessions_for_user(user_id: str, *, except_token: str | None = None) -> int:
    with _connect() as conn:
        if except_token:
            cursor = conn.execute(
                "DELETE FROM auth_sessions WHERE user_id = ? AND token_hash <> ?",
                (user_id, hash_session_token(except_token)),
            )
        else:
            cursor = conn.execute(
                "DELETE FROM auth_sessions WHERE user_id = ?",
                (user_id,),
            )
    return cursor.rowcount


def update_user_profile(user_id: str, *, name: str | None = None) -> dict[str, Any] | None:
    user = get_user_by_id(user_id)
    if user is None:
        return None
    next_name = (name if name is not None else user["name"]).strip()
    if not next_name:
        raise ValueError("Nazwa użytkownika jest wymagana.")
    with _connect() as conn:
        conn.execute(
            """
            UPDATE users
            SET name = ?, updated_at_utc = ?
            WHERE id = ?
            """,
            (next_name, utc_iso(), user_id),
        )
    return get_user_by_id(user_id)


def create_password_reset_token_for_email(
    email: str,
    *,
    ttl_minutes: int = 60,
) -> tuple[str, dict[str, Any]] | None:
    clean_email = normalize_email(email)
    with _connect() as conn:
        row = conn.execute("SELECT id FROM users WHERE email = ?", (clean_email,)).fetchone()
    if row is None:
        return None
    return create_auth_token(
        row["id"],
        purpose="password_reset",
        ttl=timedelta(minutes=ttl_minutes),
    )


def reset_password_with_token(token: str, new_password: str) -> dict[str, Any] | None:
    if len(new_password) < 8:
        raise ValueError("Hasło musi mieć co najmniej 8 znaków.")

    now = utc_iso()
    token_hash = hash_session_token(token)
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT auth_tokens.id AS token_id, users.id AS user_id
            FROM auth_tokens
            JOIN users ON users.id = auth_tokens.user_id
            WHERE auth_tokens.token_hash = ?
              AND auth_tokens.purpose = 'password_reset'
              AND auth_tokens.consumed_at_utc IS NULL
              AND auth_tokens.expires_at_utc > ?
            """,
            (token_hash, now),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE auth_tokens SET consumed_at_utc = ? WHERE id = ?",
            (now, row["token_id"]),
        )
        conn.execute(
            """
            UPDATE users
            SET password_hash = ?, updated_at_utc = ?
            WHERE id = ?
            """,
            (hash_password(new_password), now, row["user_id"]),
        )
        conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (row["user_id"],))
        user_id = row["user_id"]
    return get_user_by_id(user_id)


def create_auth_token(
    user_id: str,
    *,
    purpose: str,
    ttl: timedelta,
) -> tuple[str, dict[str, Any]]:
    token = secrets.token_urlsafe(32)
    token_hash = hash_session_token(token)
    token_id = str(uuid.uuid4())
    now = utc_now()
    expires_at = now + ttl
    with _connect() as conn:
        conn.execute(
            """
            UPDATE auth_tokens
            SET consumed_at_utc = ?
            WHERE user_id = ? AND purpose = ? AND consumed_at_utc IS NULL
            """,
            (utc_iso(now), user_id, purpose),
        )
        conn.execute(
            """
            INSERT INTO auth_tokens (
                id, user_id, purpose, token_hash, created_at_utc, expires_at_utc, consumed_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, NULL)
            """,
            (token_id, user_id, purpose, token_hash, utc_iso(now), utc_iso(expires_at)),
        )
    return token, {
        "id": token_id,
        "purpose": purpose,
        "expiresAtUtc": utc_iso(expires_at),
    }


def create_project(owner_user_id: str, *, name: str, description: str = "") -> dict[str, Any]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Nazwa projektu jest wymagana.")
    project_id = str(uuid.uuid4())
    now = utc_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO projects (
                id, owner_user_id, name, description, created_at_utc, updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (project_id, owner_user_id, clean_name, description.strip(), now, now),
        )
    project = get_project_for_user(project_id, owner_user_id)
    if project is None:
        raise RuntimeError("Nie udało się utworzyć projektu.")
    return project


def list_projects_for_user(owner_user_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                projects.*,
                (
                    SELECT id
                    FROM project_upload_sessions
                    WHERE project_upload_sessions.project_id = projects.id
                    ORDER BY created_at_utc DESC
                    LIMIT 1
                ) AS latest_session_id,
                (
                    SELECT source_pdf
                    FROM project_upload_sessions
                    WHERE project_upload_sessions.project_id = projects.id
                    ORDER BY created_at_utc DESC
                    LIMIT 1
                ) AS latest_source_pdf,
                (
                    SELECT created_at_utc
                    FROM project_upload_sessions
                    WHERE project_upload_sessions.project_id = projects.id
                    ORDER BY created_at_utc DESC
                    LIMIT 1
                ) AS latest_upload_at_utc,
                (
                    SELECT generated_at_utc
                    FROM analysis_runs
                    WHERE analysis_runs.project_id = projects.id
                    ORDER BY generated_at_utc DESC
                    LIMIT 1
                ) AS latest_analysis_at_utc,
                (
                    SELECT COUNT(*)
                    FROM analysis_runs
                    WHERE analysis_runs.project_id = projects.id
                ) AS analysis_count
            FROM projects
            WHERE owner_user_id = ? AND archived_at_utc IS NULL
            ORDER BY updated_at_utc DESC, created_at_utc DESC
            """,
            (owner_user_id,),
        ).fetchall()
    return [project_payload(dict(row)) for row in rows]


def get_project_for_user(project_id: str, owner_user_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                projects.*,
                (
                    SELECT id
                    FROM project_upload_sessions
                    WHERE project_upload_sessions.project_id = projects.id
                    ORDER BY created_at_utc DESC
                    LIMIT 1
                ) AS latest_session_id,
                (
                    SELECT source_pdf
                    FROM project_upload_sessions
                    WHERE project_upload_sessions.project_id = projects.id
                    ORDER BY created_at_utc DESC
                    LIMIT 1
                ) AS latest_source_pdf,
                (
                    SELECT created_at_utc
                    FROM project_upload_sessions
                    WHERE project_upload_sessions.project_id = projects.id
                    ORDER BY created_at_utc DESC
                    LIMIT 1
                ) AS latest_upload_at_utc,
                (
                    SELECT generated_at_utc
                    FROM analysis_runs
                    WHERE analysis_runs.project_id = projects.id
                    ORDER BY generated_at_utc DESC
                    LIMIT 1
                ) AS latest_analysis_at_utc,
                (
                    SELECT COUNT(*)
                    FROM analysis_runs
                    WHERE analysis_runs.project_id = projects.id
                ) AS analysis_count
            FROM projects
            WHERE id = ? AND owner_user_id = ? AND archived_at_utc IS NULL
            """,
            (project_id, owner_user_id),
        ).fetchone()
    return project_payload(dict(row)) if row else None


def update_project_for_user(
    project_id: str,
    owner_user_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any] | None:
    project = get_project_for_user(project_id, owner_user_id)
    if project is None:
        return None
    next_name = (name if name is not None else project["name"]).strip()
    if not next_name:
        raise ValueError("Nazwa projektu jest wymagana.")
    next_description = (
        description if description is not None else project.get("description", "")
    ).strip()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE projects
            SET name = ?, description = ?, updated_at_utc = ?
            WHERE id = ? AND owner_user_id = ?
            """,
            (next_name, next_description, utc_iso(), project_id, owner_user_id),
        )
    return get_project_for_user(project_id, owner_user_id)


def archive_project_for_user(project_id: str, owner_user_id: str) -> bool:
    with _connect() as conn:
        cursor = conn.execute(
            """
            UPDATE projects
            SET archived_at_utc = ?, updated_at_utc = ?
            WHERE id = ? AND owner_user_id = ? AND archived_at_utc IS NULL
            """,
            (utc_iso(), utc_iso(), project_id, owner_user_id),
        )
    return cursor.rowcount > 0


def record_project_upload_session(
    *,
    session_id: str,
    project_id: str,
    source_pdf: str,
) -> None:
    now = utc_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO project_upload_sessions (
                id, project_id, source_pdf, created_at_utc, last_accessed_at_utc
            )
            VALUES (?, ?, ?, COALESCE(
                (SELECT created_at_utc FROM project_upload_sessions WHERE id = ?),
                ?
            ), ?)
            """,
            (session_id, project_id, source_pdf, session_id, now, now),
        )
        conn.execute(
            "UPDATE projects SET updated_at_utc = ? WHERE id = ?",
            (now, project_id),
        )


def project_session_exists(project_id: str, session_id: str) -> bool:
    now = utc_iso()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id FROM project_upload_sessions
            WHERE id = ? AND project_id = ?
            """,
            (session_id, project_id),
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE project_upload_sessions SET last_accessed_at_utc = ? WHERE id = ?",
                (now, session_id),
            )
    return row is not None


def record_analysis_run(
    *,
    analysis_id: str,
    project_id: str,
    session_id: str,
    source_pdf: str,
    snapshot_path: str = "",
) -> None:
    now = utc_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO analysis_runs (
                id, project_id, session_id, source_pdf, snapshot_path, generated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (analysis_id, project_id, session_id, source_pdf, snapshot_path, now),
        )
        conn.execute(
            "UPDATE projects SET updated_at_utc = ? WHERE id = ?",
            (now, project_id),
        )


def list_analysis_runs_for_project(project_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, project_id, session_id, source_pdf, snapshot_path, generated_at_utc
            FROM analysis_runs
            WHERE project_id = ?
            ORDER BY generated_at_utc DESC
            """,
            (project_id,),
        ).fetchall()
    return [analysis_run_payload(dict(row)) for row in rows]


def get_analysis_run_for_project(project_id: str, analysis_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, project_id, session_id, source_pdf, snapshot_path, generated_at_utc
            FROM analysis_runs
            WHERE project_id = ? AND id = ?
            """,
            (project_id, analysis_id),
        ).fetchone()
    return analysis_run_payload(dict(row)) if row else None


def normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return (
        f"pbkdf2_sha256${PASSWORD_ITERATIONS}$"
        f"{base64.b64encode(salt).decode('ascii')}$"
        f"{base64.b64encode(digest).decode('ascii')}"
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations_raw, salt_raw, digest_raw = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = base64.b64decode(salt_raw.encode("ascii"))
        expected = base64.b64decode(digest_raw.encode("ascii"))
    except (ValueError, TypeError):
        return False

    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return secrets.compare_digest(actual, expected)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def public_user(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "createdAtUtc": row["created_at_utc"],
    }


def auth_session_payload(
    row: dict[str, Any],
    current_token_hash: str | None = None,
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "createdAtUtc": row["created_at_utc"],
        "lastSeenAtUtc": row["last_seen_at_utc"],
        "expiresAtUtc": row["expires_at_utc"],
        "isCurrent": bool(current_token_hash and row["token_hash"] == current_token_hash),
    }


def project_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row.get("description", ""),
        "createdAtUtc": row["created_at_utc"],
        "updatedAtUtc": row["updated_at_utc"],
        "latestSessionId": row.get("latest_session_id"),
        "latestSourcePdf": row.get("latest_source_pdf"),
        "latestUploadAtUtc": row.get("latest_upload_at_utc"),
        "latestAnalysisAtUtc": row.get("latest_analysis_at_utc"),
        "analysisCount": int(row.get("analysis_count") or 0),
    }


def analysis_run_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "sessionId": row["session_id"],
        "sourcePdf": row["source_pdf"],
        "snapshotPath": row.get("snapshot_path", ""),
        "generatedAtUtc": row["generated_at_utc"],
        "hasSnapshot": bool(row.get("snapshot_path")),
    }


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return public_user(dict(row)) if row else None


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    columns = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
