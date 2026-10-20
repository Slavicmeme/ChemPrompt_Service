"""LangGraph agent with a fake OpenAI Responses client: the prediction number must come from the tool."""
import asyncio
import dataclasses
import json
from types import SimpleNamespace as NS

from app.agent import graph as agent_graph
from app.agent.observability import _NoopObs


class Item(NS):
    def to_dict(self):
        return dict(self.__dict__)


class FakeStream:
    def __init__(self, events):
        self.events = events

    def __aiter__(self):
        self._it = iter(self.events)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


REASONING = Item(type="reasoning", id="rs_1", summary=[])


class FakeClient:
    def __init__(self):
        self.calls = []
        self.responses = NS(create=self.create)

    async def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            return FakeStream([
                NS(type="response.output_item.done", item=REASONING),
                NS(type="response.output_item.done", item=Item(type="function_call", call_id="call_1", name="predict_property",
                                                               arguments='{"smiles": "CCO", "property": "ESOL"}')),
                NS(type="response.completed", response=NS(usage=NS(input_tokens=10, output_tokens=5, total_tokens=15))),
            ])
        out = [i for i in kw["input"] if i.get("type") == "function_call_output"][-1]
        value = json.loads(out["output"])["result"]["value"]
        text = f"예측 값은 {value:.3f} 입니다."
        return FakeStream([
            NS(type="response.output_text.delta", delta=text[:5]),
            NS(type="response.output_text.delta", delta=text[5:]),
            NS(type="response.output_item.done", item=Item(type="message", role="assistant", content=[{"type": "output_text", "text": text}])),
            NS(type="response.completed", response=NS(usage=None)),
        ])


def test_agent_uses_tool_result_via_responses_api(monkeypatch):
    calls = []

    def fake_run_tool(user, name, args, ctx):
        calls.append((name, json.loads(args)))
        return {"status": "SUCCEEDED", "result": {"value": -0.123456}, "model_version": "ESOL:v1"}

    monkeypatch.setattr(agent_graph, "run_tool", fake_run_tool)
    monkeypatch.setattr(agent_graph, "settings", dataclasses.replace(agent_graph.settings, openai_reasoning_effort="low",
                                                                     openai_agent_model="gpt-5.6-luna"))
    events, persisted = [], []

    async def emit(ev, data):
        events.append((ev, data))

    async def persist(msg):
        persisted.append(msg)

    client = FakeClient()
    state = asyncio.run(agent_graph.GRAPH.ainvoke(
        {"messages": [{"role": "user", "content": "에탄올 ESOL 예측"}], "rounds": 0},
        config={"configurable": {"emit": emit, "client": client, "obs": _NoopObs(), "user": NS(id=1), "persist": persist,
                                 "ctx": {}, "system_prompt": agent_graph.SYSTEM_PROMPT, "prompt_version": "test"}}))

    assert calls == [("predict_property", {"smiles": "CCO", "property": "ESOL"})]
    assert state["messages"][-1]["content"] == "예측 값은 -0.123 입니다."
    first, second = client.calls
    assert first["reasoning"] == {"effort": "low"} and first["model"] == "gpt-5.6-luna"
    assert first["instructions"] == agent_graph.SYSTEM_PROMPT and first["stream"] is True
    assert first["tools"][0]["type"] == "function" and "function" not in first["tools"][0]   # Responses tool format
    # second call echoes the reasoning item + function_call, then the tool output
    kinds = [i.get("type") or i.get("role") for i in second["input"]]
    assert kinds == ["user", "reasoning", "function_call", "function_call_output"]
    ev_kinds = [e for e, _ in events]
    assert ev_kinds.index("tool_start") < ev_kinds.index("tool_result") < ev_kinds.index("token")
    assert [m["role"] for m in persisted] == ["assistant", "tool", "assistant"]
    assert all("_items" not in m for m in persisted)


def test_history_rebuilt_as_responses_items():
    hist = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "list_properties", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        {"role": "assistant", "content": "answer"},
    ]
    items = agent_graph.to_responses_input(hist)
    assert items == [
        {"role": "user", "content": "q"},
        {"type": "function_call", "call_id": "c1", "name": "list_properties", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "{}"},
        {"role": "assistant", "content": "answer"},
    ]


def test_reasoning_param_omitted_when_empty(monkeypatch):
    monkeypatch.setattr(agent_graph, "settings", dataclasses.replace(agent_graph.settings, openai_reasoning_effort=""))

    class OneShot(FakeClient):
        async def create(self, **kw):
            self.calls.append(kw)
            return FakeStream([NS(type="response.output_text.delta", delta="hi"),
                               NS(type="response.completed", response=NS(usage=None))])

    async def noop(*_):
        pass

    client = OneShot()
    asyncio.run(agent_graph.GRAPH.ainvoke(
        {"messages": [{"role": "user", "content": "hi"}], "rounds": 0},
        config={"configurable": {"emit": noop, "client": client, "obs": _NoopObs(), "user": NS(id=1), "persist": noop,
                                 "ctx": {}, "system_prompt": "s", "prompt_version": "t"}}))
    assert "reasoning" not in client.calls[0]
