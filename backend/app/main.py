"""ChemPrompt Web Service — FastAPI entrypoint."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import jobs
from app.config import PROJECT_ROOT, settings
from app.routers import admin, agent, auth, datasets, models_api, predictions, properties, training, workflows
from app.services.representation import current_info

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.secret_key:
        logging.warning("SECRET_KEY is empty — set it in .env for any non-local deployment")
    jobs.recover()
    yield


app = FastAPI(title="ChemPrompt Web Service", version="0.1.0", lifespan=lifespan)
for r in (auth, properties, predictions, datasets, workflows, training, models_api, admin, agent):
    app.include_router(r.router)


@app.get("/api/health")
def health():
    info = current_info()
    return {"status": "ok", "representation": info.label, "representation_mode": settings.representation_mode,
            "ga_mode": settings.ga_mode, "agent_configured": bool(settings.openai_api_key),
            "langfuse_configured": bool(settings.langfuse_public_key and settings.langfuse_secret_key)}


# Local development convenience; in Docker, Nginx serves ./frontend and proxies /api.
_frontend = os.environ.get("FRONTEND_DIR", str(PROJECT_ROOT / "frontend"))
if os.environ.get("SERVE_FRONTEND", "false").lower() == "true" and os.path.isdir(_frontend):
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
