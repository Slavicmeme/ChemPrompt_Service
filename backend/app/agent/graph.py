"""LangGraph orchestration for the chat agent (docs/12, docs/14, docs/15).

    START -> agent (OpenAI Responses API, streamed) --function calls--> tools -> agent ... --none--> END

* OpenAI (ChatGPT) understands the request, extracts parameters, asks for missing input, calls tools
  and explains results. It never produces prediction values: the system prompt forbids it and the
  only path to a number is the ``predict_property`` tool (prediction_service).
* LangGraph is only the in-flight orchestration. Persistent state (messages, predictions, workflows)
  is written to PostgreSQL.
* Every SSE event corresponds to a real action (model call, tool call, token from OpenAI).
"""
from __future__ import annotations

import asyncio
import json
import operator
import time
from typing import Annotated, Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from app.agent.observability import observer
from app.agent.tools import TOOL_SPECS, run_tool
from app.config import settings

MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """당신은 ChemPrompt 분자 물성 예측 서비스의 연구 보조 Agent입니다. 사용자는 화학·생물·약학 연구자입니다.

규칙:
1. 예측 값(수치, 확률, label)을 절대 직접 만들거나 추정하지 마십시오. 예측은 반드시 predict_property tool의 결과만 사용합니다.
   tool 결과에 없는 숫자를 말하지 마십시오. tool이 error를 반환하면 그대로 설명하고 값을 지어내지 마십시오.
2. 예측에 필요한 정보(SMILES, property)가 없으면 먼저 질문하십시오. 분자 이름만 주어지면 SMILES를 확인받으십시오.
3. 결과 설명 시: 예측 값과 단위, Model 1~5 예측의 모델 간 표준편차(불확실성의 참고치), 모델 버전,
   descriptor mode, 데이터 분할 방식과 검증 성능을 함께 알려주십시오. 분류는 확률과 threshold를 설명하십시오.
   사용자에게는 fold, seed 같은 내부 용어 대신 "Model 1~5"로 설명하십시오(예: "Model 1~5의 평균").
   property는 국문 이름과 함께 말하십시오(예: "ESOL(수용해도)"). tool 결과의 property.name_ko / target_info를 사용하십시오.
   분류 결과는 코드명(FDA_APPROVED, CT_TOX)만 쓰지 말고 target_info의 뜻(예: "FDA 승인 여부: 승인 약물일 확률 0.97")으로 설명하십시오.
   검증 성능을 말할 때는 "학습에 쓰지 않은 데이터로 평가한 성능"이라는 뜻과 각 지표의 의미를 한 줄로 덧붙이십시오.
4. 학습/워크플로 상태는 get_training_status / get_workflow_status로 확인한 사실만 말하십시오.
5. 한국어로 간결하고 정확하게 답하십시오. 이 예측은 연구 참고용이며 실험 검증을 대체하지 않는다는 점을 필요 시 언급하십시오.
"""

Emit = Callable[[str, dict], Awaitable[None]]


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    rounds: int


RESPONSES_TOOLS = [
    # Responses API tool format. strict=False because some parameters are optional.
    {"type": "function", "name": t["function"]["name"], "description": t["function"]["description"],
     "parameters": t["function"]["parameters"], "strict": False}
    for t in TOOL_SPECS
]


def _item_dict(item: Any) -> dict:
    return item.to_dict() if hasattr(item, "to_dict") else item.model_dump(exclude_none=True, by_alias=True)


def to_responses_input(messages: list[dict]) -> list[dict]:
    """Our stored chat-style messages -> Responses API input items.

    Assistant turns produced in this run carry the raw output items (``_items``: reasoning, message,
    function_call) so they are echoed back exactly, as the Responses API expects for reasoning models.
    Turns loaded from the DB are rebuilt from text and tool calls.
    """
    items: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            items.append({"role": "user", "content": m.get("content") or ""})
        elif role == "assistant":
            if m.get("_items"):
                items.extend(m["_items"])
                continue
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
            for tc in m.get("tool_calls") or []:
                items.append({"type": "function_call", "call_id": tc["id"], "name": tc["function"]["name"],
                              "arguments": tc["function"]["arguments"]})
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m.get("content") or ""})
    return items


async def _agent_node(state: AgentState, config) -> dict:
    c = config["configurable"]
    emit: Emit = c["emit"]
    client: AsyncOpenAI = c["client"]
    obs = c["obs"]
    input_items = to_responses_input(state["messages"])
    allow_tools = state["rounds"] < MAX_TOOL_ROUNDS
    await emit("status", {"state": "model_call", "model": settings.openai_agent_model})
    kwargs: dict[str, Any] = {"model": settings.openai_agent_model, "instructions": c["system_prompt"],
                              "input": input_items, "stream": True}
    if allow_tools:
        kwargs["tools"] = RESPONSES_TOOLS
    if settings.openai_reasoning_effort:
        kwargs["reasoning"] = {"effort": settings.openai_reasoning_effort}
    gen = obs.start(name="openai.responses", as_type="generation", model=settings.openai_agent_model,
                    input=input_items, model_parameters={"reasoning_effort": settings.openai_reasoning_effort or None},
                    metadata={"prompt_version": c["prompt_version"], "round": state["rounds"]})
    t0 = time.time()
    content: list[str] = []
    out_items: list[dict] = []
    calls: list[dict] = []
    usage = None
    try:
        stream = await client.responses.create(**kwargs)
        async for ev in stream:
            et = getattr(ev, "type", "")
            if et == "response.output_text.delta":
                content.append(ev.delta)
                await emit("token", {"text": ev.delta})
            elif et == "response.output_item.done":
                item = ev.item
                out_items.append(_item_dict(item))
                if getattr(item, "type", "") == "function_call":
                    calls.append({"id": item.call_id, "type": "function",
                                  "function": {"name": item.name, "arguments": item.arguments or "{}"}})
            elif et == "response.completed":
                usage = getattr(ev.response, "usage", None)
            elif et == "response.failed":
                err = getattr(ev.response, "error", None)
                raise RuntimeError(f"OpenAI response failed: {getattr(err, 'message', err)}")
            elif et == "error":
                raise RuntimeError(f"OpenAI stream error: {getattr(ev, 'message', ev)}")
    except Exception as exc:
        gen.update(level="ERROR", status_message=str(exc))
        gen.end()
        raise
    msg: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None, "_items": out_items}
    if calls:
        msg["tool_calls"] = calls
    usage_details = None
    if usage is not None:
        usage_details = {"input": usage.input_tokens, "output": usage.output_tokens, "total": usage.total_tokens}
    gen.update(output={"content": msg["content"], "tool_calls": calls}, usage_details=usage_details,
               metadata={"latency_ms": int((time.time() - t0) * 1000)})
    gen.end()
    await c["persist"]({k: v for k, v in msg.items() if k != "_items"})
    return {"messages": [msg]}


async def _tools_node(state: AgentState, config) -> dict:
    c = config["configurable"]
    emit: Emit = c["emit"]
    obs = c["obs"]
    out = []
    for call in state["messages"][-1].get("tool_calls", []):
        name, args = call["function"]["name"], call["function"]["arguments"]
        await emit("tool_start", {"id": call["id"], "name": name, "arguments": _safe_json(args)})
        span = obs.start(name=f"tool:{name}", as_type="tool", input=_safe_json(args))
        t0 = time.time()
        result = await asyncio.to_thread(run_tool, c["user"], name, args, c["ctx"])
        span.update(output=result, level="ERROR" if isinstance(result, dict) and "error" in result else "DEFAULT",
                    metadata={"latency_ms": int((time.time() - t0) * 1000)})
        span.end()
        await emit("tool_result", {"id": call["id"], "name": name, "result": result})
        if name == "get_workflow_status" and isinstance(result, dict) and "current_step" in result:
            await emit("workflow_step", {"workflow_id": result["workflow_id"], "current_step": result["current_step"],
                                         "status": result["status"]})
        msg = {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False, default=str)}
        await c["persist"](msg | {"_tool_name": name})
        out.append(msg)
    return {"messages": out, "rounds": state["rounds"] + 1}


def _route(state: AgentState) -> str:
    return "tools" if state["messages"][-1].get("tool_calls") else END


def _safe_json(s: str) -> Any:
    try:
        return json.loads(s or "{}")
    except json.JSONDecodeError:
        return s


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", _agent_node)
    g.add_node("tools", _tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", _route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


GRAPH = build_graph()
