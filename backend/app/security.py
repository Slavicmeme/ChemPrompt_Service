"""Auth (docs/02_AUTH.md): Argon2id password hashes + opaque server-side session tokens."""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import User, UserSession

_ph = PasswordHasher()  # argon2-cffi default = Argon2id
COOKIE_NAME = "cp_session"


def hash_password(pw: str) -> str:
    return _ph.hash(pw)


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return _ph.verify(hashed, pw)
    except (VerificationError, InvalidHashError):
        return False


def _token_hash(token: str) -> str:
    material = f"{settings.secret_key}:{token}" if settings.secret_key else token
    return hashlib.sha256(material.encode()).hexdigest()


def create_session(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(32)
    db.add(UserSession(token_hash=_token_hash(token), user_id=user.id,
                       expires_at=datetime.now(timezone.utc) + timedelta(hours=settings.session_ttl_hours)))
    db.flush()
    return token


def resolve_session(db: Session, token: str | None) -> User | None:
    if not token:
        return None
    s = db.scalar(select(UserSession).where(UserSession.token_hash == _token_hash(token)))
    if s is None or s.expires_at < datetime.now(timezone.utc) or not s.user.is_active:
        return None
    return s.user


def destroy_session(db: Session, token: str | None) -> None:
    if token:
        db.execute(delete(UserSession).where(UserSession.token_hash == _token_hash(token)))
