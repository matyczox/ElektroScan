"""Authentication helpers shared by API routes."""

from __future__ import annotations

import os

from fastapi import HTTPException, Request, Response

from auth_store import get_user_by_session_token


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


SESSION_COOKIE_NAME = "elektroscan_session"
SESSION_TTL_DAYS = int(os.getenv("ELEKTROSCAN_SESSION_TTL_DAYS", "30"))
AUTH_COOKIE_SECURE = str(os.getenv("ELEKTROSCAN_AUTH_COOKIE_SECURE", "")).lower() in {
    "1",
    "true",
    "yes",
    "on",
}
AUTH_DEV_TOKENS = _env_flag("ELEKTROSCAN_AUTH_DEV_TOKENS", default=True)


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=SESSION_TTL_DAYS * 24 * 60 * 60,
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def _clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")


def _current_user_from_request(request: Request) -> dict | None:
    return get_user_by_session_token(request.cookies.get(SESSION_COOKIE_NAME))


def _dev_auth_token_payload(field_name: str, token: str | None) -> dict:
    if not AUTH_DEV_TOKENS or not token:
        return {}
    return {field_name: token}


def require_user(request: Request) -> dict:
    user = _current_user_from_request(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Wymagane logowanie.")
    return user
