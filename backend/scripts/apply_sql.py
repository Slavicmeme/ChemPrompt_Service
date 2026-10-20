"""Apply ./sql/*.sql in order (idempotent DDL). Docker's postgres image does this automatically on first init."""
from __future__ import annotations

from pathlib import Path

import psycopg

from app.config import PROJECT_ROOT, settings


def main() -> None:
    url = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(url, autocommit=True) as conn:
        for f in sorted(Path(PROJECT_ROOT / "sql").glob("*.sql")):
            conn.execute(f.read_text())
            print(f"applied {f.name}")


if __name__ == "__main__":
    main()
