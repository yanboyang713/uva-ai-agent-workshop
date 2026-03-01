"""
Task 5: Long-running conversation with LangGraph + checkpoint/recovery.

Graph structure:
Human input -> agent -> (tools?) -> agent -> ... -> final assistant response
"""

import argparse
import ast
import json
import math
import os
from typing import Annotated, Any, TypedDict

from langchain.tools import tool
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


def area_circle(r: float) -> float:
    return math.pi * r * r


def circumference_circle(r: float) -> float:
    return 2 * math.pi * r


def area_rectangle(width: float, height: float) -> float:
    return width * height


def area_triangle(base: float, height: float) -> float:
    return 0.5 * base * height


ALLOWED_CALC_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": pow,
    "abs": abs,
    "round": round,
    "area_circle": area_circle,
    "circumference_circle": circumference_circle,
    "area_rectangle": area_rectangle,
    "area_triangle": area_triangle,
}


ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.UAdd,
    ast.USub,
)


def _validate_expression(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported syntax in expression: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_CALC_NAMES:
            raise ValueError(f"Unknown symbol: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed")
            if node.func.id not in ALLOWED_CALC_NAMES:
                raise ValueError(f"Function not allowed: {node.func.id}")


def safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    _validate_expression(tree)
    result = eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, ALLOWED_CALC_NAMES)
    if not isinstance(result, (int, float)):
        raise ValueError("Expression did not produce a numeric result")
    return float(result)


@tool
def calculator(expression: str, precision: int = 6) -> str:
    """Evaluate a math expression with trig/geometry helper functions."""
    try:
        value = safe_eval(expression)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "expression": expression, "error": str(exc)})
    return json.dumps({"ok": True, "expression": expression, "result": round(value, precision)})


@tool
def count_letter(text: str, letter: str, case_sensitive: bool = False) -> str:
    """Count occurrences of one letter in text."""
    if len(letter) != 1:
        return json.dumps({"ok": False, "error": "letter must be a single character"})
    haystack = text if case_sensitive else text.lower()
    needle = letter if case_sensitive else letter.lower()
    return json.dumps({"ok": True, "count": haystack.count(needle), "text": text, "letter": letter})


@tool
def text_stats(text: str) -> str:
    """Return text stats: characters, words, and unique letters."""
    words = [w for w in text.split() if w]
    unique_letters = sorted({ch.lower() for ch in text if ch.isalpha()})
    return json.dumps(
        {
            "ok": True,
            "chars": len(text),
            "words": len(words),
            "unique_letters": unique_letters,
            "unique_letter_count": len(unique_letters),
        }
    )


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    weather_data = {
        "san francisco": "Sunny, 72F",
        "new york": "Cloudy, 55F",
        "london": "Rainy, 48F",
        "tokyo": "Clear, 65F",
        "charlottesville": "Partly cloudy, 68F",
    }
    forecast = weather_data.get(location.lower(), f"Weather data not available for {location}")
    return json.dumps({"location": location, "forecast": forecast})


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(model: str):
    system_msg = SystemMessage(
        content=(
            "You are a helpful assistant. Use tools for weather, counting, text stats, and math. "
            "Prefer tool calls over mental math when calculations/counts are requested."
        )
    )
    tools = [get_weather, calculator, count_letter, text_stats]
    tool_map = {tool_obj.name: tool_obj for tool_obj in tools}
    llm_with_tools = ChatOpenAI(model=model, temperature=0).bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        prompt_messages: list[AnyMessage]
        if any(isinstance(msg, SystemMessage) for msg in state["messages"]):
            prompt_messages = state["messages"]
        else:
            prompt_messages = [system_msg, *state["messages"]]
        response = llm_with_tools.invoke(prompt_messages)
        return {"messages": [response]}

    def call_tools(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": tool_messages}

        for tool_call in last_message.tool_calls:
            function_name = tool_call["name"]
            function_args = tool_call["args"]
            if function_name in tool_map:
                result = tool_map[function_name].invoke(function_args)
            else:
                result = json.dumps({"ok": False, "error": f"Unknown function {function_name}"})
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=function_name,
                )
            )
        return {"messages": tool_messages}

    def route_after_model(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", call_tools)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_model, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder


def _print_recent_history(messages: list[AnyMessage], limit: int = 8) -> None:
    print("\nRecent history:")
    for msg in messages[-limit:]:
        role = type(msg).__name__
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"[{role}] tool_calls={len(msg.tool_calls)}")
        else:
            print(f"[{role}] {content}")


def _print_last_assistant(messages: list[AnyMessage]) -> None:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            print(f"\nAssistant: {content}\n")
            return
    print("\nAssistant: (no final assistant message found)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5 LangGraph checkpointed conversation.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--thread-id", default=os.environ.get("LANGGRAPH_THREAD_ID", "topic3-tools-chat"))
    parser.add_argument(
        "--checkpoint-db",
        default=os.path.join(os.path.dirname(__file__), "outputs", "task5", "task5_checkpoints.db"),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Optional non-interactive mode. Repeat --prompt to run multiple turns and exit.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    builder = build_graph(model=args.model)
    config = {"configurable": {"thread_id": args.thread_id}}

    with SqliteSaver.from_conn_string(args.checkpoint_db) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        try:
            saved = graph.get_state(config)
            saved_messages = saved.values.get("messages", []) if saved and saved.values else []
        except Exception:  # noqa: BLE001
            saved_messages = []

        if saved_messages:
            print(f"Recovered thread '{args.thread_id}' with {len(saved_messages)} message(s).")
            _print_recent_history(saved_messages)
        else:
            print(f"Started new thread '{args.thread_id}'.")

        if args.prompt:
            for prompt in args.prompt:
                print(f"\nUser: {prompt}")
                state = graph.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
                _print_last_assistant(state["messages"])
            return

        print("\nType '/history' to print recent context and '/quit' to exit.")
        while True:
            user_text = input("\n> ").strip()
            if user_text.lower() in {"/quit", "quit", "exit", "q"}:
                print("Goodbye.")
                break
            if user_text.lower() == "/history":
                current = graph.get_state(config)
                msgs = current.values.get("messages", []) if current and current.values else []
                _print_recent_history(msgs)
                continue
            if not user_text:
                continue

            state = graph.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)
            _print_last_assistant(state["messages"])


if __name__ == "__main__":
    main()
