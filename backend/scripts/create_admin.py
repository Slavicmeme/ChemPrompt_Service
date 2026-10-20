"""Create (or promote) an ADMIN user.  python -m scripts.create_admin <username> [--password ...]"""
from __future__ import annotations

import argparse
import getpass

from sqlalchemy import select

from app.db import session_scope
from app.models import User
from app.security import hash_password


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("username")
    ap.add_argument("--password", help="omit to be prompted")
    a = ap.parse_args()
    pw = a.password
    if pw is None:
        pw = getpass.getpass("password: ")
        if getpass.getpass("password (again): ") != pw:
            raise SystemExit("passwords do not match — admin not created")
    if len(pw) < 8:
        raise SystemExit("password must be >= 8 chars — admin not created")
    with session_scope() as db:
        u = db.scalar(select(User).where(User.username == a.username))
        if u is None:
            db.add(User(username=a.username, password_hash=hash_password(pw), role="ADMIN"))
            print(f"created admin {a.username}")
        else:
            u.role, u.password_hash = "ADMIN", hash_password(pw)
            print(f"promoted/updated {a.username} to ADMIN")


if __name__ == "__main__":
    main()
