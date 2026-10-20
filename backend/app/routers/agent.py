from __future__ import annotations

import asyncio
import hashlib

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.graph import GRAPH, SYSTEM_PROMPT
from app.agent.observability import observer
from app.config import settings
from app.db import get_db, session_scope
from app.deps import current_user
from app.models import AgentConversation, AgentMessage, User
from app.routers.common import SSE_HEADERS, sse

router = APIRouter(prefix="/api/agent", tags=["agent"])
HISTORY_LIMIT = 40


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: int | None = None


def _history(db: Session, conv_id: int) -> list[dict]:
    rows = db.scalars(select(AgentMessage).where(AgentMessage.conversation_id == conv_id)
                      .order_by(AgentMessage.id.desc()).limit(HISTORY_LIMIT)).all()[::-1]
    msgs = []
    for m in rows:
        if m.role == "assistant":
            d = {"role": "assistant", "content": m.content}
            if m.tool_calls:
                d["tool_calls"] = m.tool_calls
            msgs.append(d)
        elif m.role == "tool":
            msgs.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or ""})
        else:
            msgs.append({"role": "user", "content": m.content or ""})
    while msgs and msgs[0]["role"] == "tool":  # never start with an orphaned tool result
        msgs.pop(0)
    return msgs


def _user_ref(user: User) -> str:
    """Internal pseudonymous reference for Langfuse (no username)."""
    return "u_" + hashlib.sha256(f"{settings.secret_key}:{user.id}".encode()).hexdigest()[:16]


@router.post("/chat")
async def chat(body: ChatRequest, db: Session = Depends(get_db), user: User = Depends(current_user)):
    if body.conversation_id:
        conv = db.get(AgentConversation, body.conversation_id)
        if conv is None or conv.user_id != user.id:
            raise HTTPException(404, "conversation not found")
    else:
        conv = AgentConversation(user_id=user.id, title=body.message[:80])
        db.add(conv)
        db.flush()
    history = _history(db, conv.id)
    db.add(AgentMessage(conversation_id=conv.id, role="user", content=body.message))
    db.commit()
    conv_id = conv.id

    queue: asyncio.Queue = asyncio.Queue()

    async def emit(event: str, data: dict) -> None:
        await queue.put((event, data))

    async def persist(msg: dict) -> None:
        def _save():
            with session_scope() as s:
                s.add(AgentMessage(conversation_id=conv_id, role=msg["role"], content=msg.get("content"),
                                   tool_calls=msg.get("tool_calls"), tool_call_id=msg.get("tool_call_id"),
                                   tool_name=msg.get("_tool_name")))
        await asyncio.to_thread(_save)

    async def run() -> None:
        if not settings.openai_api_key:
            await emit("error", {"error": "OPENAI_API_KEY가 설정되지 않았습니다. 관리자에게 문의하세요."})
            return
        from openai import AsyncOpenAI

        system_prompt, prompt_version = observer.system_prompt(SYSTEM_PROMPT)
        obs = observer.trace(name="agent-chat", user_ref=_user_ref(user), session_id=f"conv-{conv_id}",
                             input=body.message, metadata={"conversation_id": conv_id, "prompt_version": prompt_version})
        try:
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            state = await GRAPH.ainvoke(
                {"messages": [*history, {"role": "user", "content": body.message}], "rounds": 0},
                config={"configurable": {"emit": emit, "client": client, "obs": obs, "user": user, "persist": persist,
                                         "ctx": {"conversation_id": conv_id}, "system_prompt": system_prompt,
                                         "prompt_version": prompt_version},
                        "recursion_limit": 20},
            )
            final = state["messages"][-1]
            obs.update(output=final.get("content"))
        except Exception as exc:  # noqa: BLE001
            obs.update(level="ERROR", status_message=str(exc))
            await emit("error", {"error": f"Agent 처리 중 오류: {exc}"})
        finally:
            obs.end()
            observer.flush()

    async def gen():
        yield sse("status", {"state": "started", "conversation_id": conv_id})
        task = asyncio.create_task(run())
        task.add_done_callback(lambda _: queue.put_nowait(("__end__", {})))
        while True:
            event, data = await queue.get()
            if event == "__end__":
                break
            yield sse(event, data)
        yield sse("done", {"conversation_id": conv_id})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.get("/conversations")
def conversations(db: Session = Depends(get_db), user: User = Depends(current_user)):
    rows = db.scalars(select(AgentConversation).where(AgentConversation.user_id == user.id)
                      .order_by(AgentConversation.id.desc()).limit(50)).all()
    return {"conversations": [{"id": c.id, "title": c.title, "created_at": c.created_at} for c in rows]}


@router.get("/conversations/{conv_id}")
def conversation(conv_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    c = db.get(AgentConversation, conv_id)
    if c is None or c.user_id != user.id:
        raise HTTPException(404, "conversation not found")
    return {"id": c.id, "title": c.title,
            "messages": [{"role": m.role, "content": m.content, "tool_calls": m.tool_calls, "tool_name": m.tool_name,
                          "created_at": m.created_at} for m in c.messages]}
