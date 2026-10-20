from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User
from app.security import COOKIE_NAME, resolve_session


def session_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.cookies.get(COOKIE_NAME)


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = resolve_session(db, session_token(request))
    if user is None:
        raise HTTPException(401, "로그인이 필요합니다.")
    return user


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "ADMIN":  # role is always checked server-side
        raise HTTPException(403, "관리자 권한이 필요합니다.")
    return user
