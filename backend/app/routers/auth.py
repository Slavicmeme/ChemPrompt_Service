from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.deps import current_user, session_token
from app.models import User
from app.security import COOKIE_NAME, create_session, destroy_session, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])
_USERNAME = re.compile(r"^[A-Za-z0-9_.\-]{3,64}$")


class Credentials(BaseModel):
    username: str
    password: str


def user_dict(u: User) -> dict:
    return {"id": u.id, "username": u.username, "role": u.role}


def _set_cookie(resp: Response, token: str) -> None:
    resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax", secure=settings.cookie_secure,
                    max_age=settings.session_ttl_hours * 3600, path="/")


@router.post("/register")
def register(body: Credentials, response: Response, db: Session = Depends(get_db)):
    if not _USERNAME.match(body.username):
        raise HTTPException(400, "username은 3~64자의 영문/숫자/_.- 만 가능합니다.")
    if len(body.password) < 8:
        raise HTTPException(400, "password는 8자 이상이어야 합니다.")
    if db.scalar(select(User).where(User.username == body.username)):
        raise HTTPException(409, "이미 존재하는 username입니다.")
    u = User(username=body.username, password_hash=hash_password(body.password), role="USER")
    db.add(u)
    db.flush()
    token = create_session(db, u)
    db.commit()
    _set_cookie(response, token)
    return {"user": user_dict(u)}


@router.post("/login")
def login(body: Credentials, response: Response, db: Session = Depends(get_db)):
    u = db.scalar(select(User).where(User.username == body.username))
    if u is None or not u.is_active or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "username 또는 password가 올바르지 않습니다.")
    token = create_session(db, u)
    db.commit()
    _set_cookie(response, token)
    return {"user": user_dict(u)}  # token only travels in the HttpOnly cookie


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    destroy_session(db, session_token(request))
    db.commit()
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user)):
    return {"user": user_dict(user)}
