"""Agent/tool-use benchmark — multi-turn tool loops with simulated execution.

Extends function_calling with a full agent loop: model calls tool → tool is
executed (simulated) → result fed back → repeat until model produces final
answer. Scores both correct tool orchestration and final answer correctness.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

from local_ai_bench.domain.models import (
    BenchmarkCase,
    BenchmarkCategory,
    CaseResult,
    Evaluation,
    Modality,
    ModelInfo,
    PluginAggregate,
)
from local_ai_bench.plugins.base import MultiTurnPlugin as MultiTurnCapability
from local_ai_bench.plugins.base import RunContext
from local_ai_bench.plugins.builtin._base import BaseTextPlugin
from local_ai_bench.plugins.score import normalize_text

# Tool implementations — pure functions the "agent" can call
# In a real deployment these would call external APIs; here they're pure for
# deterministic scoring. The `calculate` tool deliberately uses a safe
# AST evaluator (never `eval`) because the expression comes from the model.
TOOL_IMPLS = {
    "get_weather": lambda city: (
        f"Weather in {city}: 22°C, partly cloudy, humidity 65%"
    ),
    "calculate": lambda expression: f"Result: {_safe_calculate(expression)}",
    "lookup_user": lambda user_id: (
        f"User {user_id}: name=Alex Chen, email=alex@example.com, tier=premium"
    ),
    "query_orders": lambda user_id: (
        f"Orders for {user_id}: #1001 ($49.99, shipped), #1002 ($12.50, pending)"
    ),
    "check_inventory": lambda sku: (
        f"SKU {sku}: 42 units in stock, reorder point 10"
    ),
}


def _safe_calculate(expression: str) -> float:
    """Evaluate a numeric expression using literals and arithmetic ops only.

    Interprets the parsed AST directly (no ``eval``) so a model-supplied
    expression can never execute arbitrary code. Anything beyond numbers,
    arithmetic operators, and parentheses is rejected.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid arithmetic expression: {expression!r}") from exc

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left**right
        if isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
        raise ValueError(f"unsupported syntax: {type(node).__name__}")

    return float(_eval_node(tree.body))


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Normalize Ollama tool arguments (dict or JSON string) to a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}

# Each case: initial prompt, available tools, expected final answer,
# and required tool-call sequence (for orchestration scoring)
_CASES = [
    {
        "id": "agent_weather_0001",
        "prompt": "What's the weather in Tokyo right now?",
        "tools": ["get_weather"],
        "expected_answer": "Tokyo",
        "expected_tool_sequence": [("get_weather", {"city": "Tokyo"})],
        "answer_keywords": ["22", "celsius", "partly cloudy", "65"],
    },
    {
        "id": "agent_math_0002",
        "prompt": "Calculate (15 * 24) + (360 / 12) and tell me the result.",
        "tools": ["calculate"],
        "expected_answer": "420",
        "expected_tool_sequence": [("calculate", {"expression": "(15 * 24) + (360 / 12)"})],
        "answer_keywords": ["420", "result"],
    },
    {
        "id": "agent_user_lookup_0003",
        "prompt": "Look up user U-42 and tell me their email address.",
        "tools": ["lookup_user"],
        "expected_answer": "alex@example.com",
        "expected_tool_sequence": [("lookup_user", {"user_id": "U-42"})],
        "answer_keywords": ["alex", "example.com", "email"],
    },
    {
        "id": "agent_multi_step_0004",
        "prompt": (
            "User U-42 wants to know their order history. First look them up, "
            "then query their orders. Summarize what you find."
        ),
        "tools": ["lookup_user", "query_orders"],
        "expected_answer": "U-42",
        "expected_tool_sequence": [
            ("lookup_user", {"user_id": "U-42"}),
            ("query_orders", {"user_id": "U-42"}),
        ],
        "answer_keywords": ["order", "1001", "1002", "shipped", "pending"],
    },
    {
        "id": "agent_inventory_0005",
        "prompt": "Check inventory for SKU ABC-123 and tell me if we need to reorder (reorder point is 10).",
        "tools": ["check_inventory"],
        "expected_answer": "no",
        "expected_tool_sequence": [("check_inventory", {"sku": "ABC-123"})],
        "answer_keywords": ["42", "stock", "reorder", "no"],
    },
]


def _make_tool_schema(name: str) -> dict[str, Any]:
    """Ollama tool schema for a built-in tool."""
    schemas = {
        "get_weather": {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            },
        },
        "calculate": {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"],
                },
            },
        },
        "lookup_user": {
            "type": "function",
            "function": {
                "name": "lookup_user",
                "description": "Look up user profile by ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User account ID"}
                    },
                    "required": ["user_id"],
                },
            },
        },
        "query_orders": {
            "type": "function",
            "function": {
                "name": "query_orders",
                "description": "Query order history for a user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User account ID"}
                    },
                    "required": ["user_id"],
                },
            },
        },
        "check_inventory": {
            "type": "function",
            "function": {
                "name": "check_inventory",
                "description": "Check inventory level for a SKU.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sku": {"type": "string", "description": "Product SKU"}
                    },
                    "required": ["sku"],
                },
            },
        },
    }
    return schemas[name]


class AgentToolUsePlugin(BaseTextPlugin, MultiTurnCapability):
    id: ClassVar[str] = "agent_tool_use"
    name: ClassVar[str] = "Agent Tool Use"
    description: ClassVar[str] = "Full agent loops: multi-turn tool use with simulated execution."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.FUNCTION_CALLING
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}
    max_turns: ClassVar[int] = 6  # enough for multi-step agents

    def supports_model(self, model: ModelInfo) -> bool:
        return bool(model.supports_tools)

    def build_request(self, case: BenchmarkCase, model: ModelInfo, ctx: RunContext) -> dict[str, Any]:  # noqa: ANN001
        return self.turn_request(case, model, ctx, [])

    def cases(self, ctx) -> Iterable[BenchmarkCase]:  # noqa: ANN001
        for spec in _CASES:
            yield BenchmarkCase(
                id=spec["id"],
                plugin_id=self.id,
                dataset_version=self.dataset_version,
                input={"prompt": spec["prompt"]},
                expected={
                    "tools": spec["tools"],  # list of tool names
                    "expected_answer": spec["expected_answer"],
                    "expected_tool_sequence": spec["expected_tool_sequence"],
                    "answer_keywords": spec["answer_keywords"],
                },
            )

    def turn_request(
        self,
        case: BenchmarkCase,
        model: ModelInfo,
        ctx: RunContext,  # noqa: ANN001
        transcript: list[dict[str, Any]],
    ) -> dict[str, Any]:
        expected = case.expected or {}
        prompts: list[str] = [case.input["prompt"]] + ctx.options.get("followup_prompts", [])
        tools = [_make_tool_schema(t) for t in expected.get("tools", [])]

        # Rebuild the full conversation so Ollama sees the history: the user
        # prompt for each completed turn, the assistant reply, and — whenever
        # the model issued tool calls — the deterministic tool results fed back
        # with role "tool". The runner never reconstructs history itself.
        messages: list[dict[str, Any]] = []
        for i, reply in enumerate(transcript):
            if i < len(prompts):
                messages.append({"role": "user", "content": prompts[i]})
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": reply.get("content") or "",
            }
            calls = reply.get("tool_calls")
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            if calls:
                for call in calls:
                    messages.append({"role": "tool", "content": self._execute_tool(call)})

        idx = len(transcript)
        if idx < len(prompts):
            messages.append({"role": "user", "content": prompts[idx]})
        return {
            "messages": messages,
            "options": {"temperature": 0.0, "num_predict": 256},
            "tools": tools,
        }

    def should_stop(self, case: BenchmarkCase, response: Any, ctx: RunContext, turn: int) -> bool:
        # Stop when model produces a final answer (no tool_calls) or max_turns reached
        if turn + 1 >= self.max_turns:
            return True
        # If last assistant message has no tool_calls, it's a final answer
        tool_calls = response.tool_calls or []
        return len(tool_calls) == 0

    def _execute_tool(self, call: dict[str, Any]) -> str:
        """Execute a tool call and return its result string.

        ``call`` is an Ollama tool_calls entry (``{"function": {"name", "arguments"}}``);
        arguments may be a dict or a JSON-encoded string. Tool errors become
        tool output so the model can recover, mirroring a real agent loop.
        """
        fn = call.get("function", {})
        name = fn.get("name")
        args = _parse_arguments(fn.get("arguments"))
        impl = TOOL_IMPLS.get(name)
        if impl is None:
            return f"Error: unknown tool {name}"
        try:
            return impl(**args)
        except Exception as exc:  # noqa: BLE001 - tool errors surface to the model
            return f"Error executing {name}: {exc}"

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        expected = case.expected or {}
        transcript = ctx.transcript or []

        # 1. Tool orchestration score: did the model call the right tools in order?
        expected_sequence = expected.get("expected_tool_sequence", [])
        actual_sequence: list[tuple[str, dict]] = []
        for msg in transcript:
            for call in msg.get("tool_calls") or []:
                fn = call.get("function", {})
                actual_sequence.append((fn.get("name"), _parse_arguments(fn.get("arguments"))))

        tool_score = self._score_tool_sequence(actual_sequence, expected_sequence)

        # 2. Final answer score: check last response text for expected keywords
        answer_score = self._score_answer(response.text, expected.get("answer_keywords", []))

        # Combined score: 40% tool orchestration, 60% answer correctness
        score = round(0.4 * tool_score + 0.6 * answer_score, 4)

        metrics = {
            "tool_orchestration_score": tool_score,
            "answer_score": answer_score,
            "expected_tool_sequence": expected_sequence,
            "actual_tool_sequence": actual_sequence,
            "turns": len(ctx.transcript or []),
        }
        return Evaluation(
            score=score,
            passed=score == 1.0,
            metrics=metrics,
        )

    def _score_tool_sequence(
        self, actual: list[tuple[str, dict]], expected: list[tuple[str, dict]]
    ) -> float:
        """Score actual tool calls vs expected sequence (order + name + args)."""
        if not expected:
            return 1.0 if not actual else 0.5
        if not actual:
            return 0.0

        matched = 0
        for i, (exp_name, exp_args) in enumerate(expected):
            if i < len(actual):
                act_name, act_args = actual[i]
                name_ok = act_name == exp_name
                args_ok = self._args_match(act_args, exp_args)
                if name_ok and args_ok:
                    matched += 1
                elif name_ok:
                    matched += 0.5
        return round(matched / len(expected), 4)

    def _args_match(self, actual: dict, expected: dict) -> bool:
        """Loose argument matching: keys present, values equal or numeric close."""
        for key, exp_val in expected.items():
            act_val = actual.get(key)
            if act_val is None:
                return False
            try:
                if abs(float(act_val) - float(exp_val)) > 0.5:
                    return False
            except (TypeError, ValueError):
                if normalize_text(str(act_val)) != normalize_text(str(exp_val)):
                    return False
        return True

    def _score_answer(self, text: str, keywords: list[str]) -> float:
        """Keyword recall for final answer."""
        if not keywords:
            return 1.0
        norm = normalize_text(text)
        hits = sum(1 for kw in keywords if normalize_text(kw) in norm)
        return round(hits / len(keywords), 4)

    def aggregate(self, results: Sequence[CaseResult]) -> PluginAggregate:
        agg = super().aggregate(results)
        tool_scores = [r.evaluation.metrics.get("tool_orchestration_score", 0.0) for r in results]
        answer_scores = [r.evaluation.metrics.get("answer_score", 0.0) for r in results]
        agg.metrics["avg_tool_score"] = round(sum(tool_scores) / len(results), 4) if results else 0.0
        agg.metrics["avg_answer_score"] = round(sum(answer_scores) / len(results), 4) if results else 0.0
        return agg