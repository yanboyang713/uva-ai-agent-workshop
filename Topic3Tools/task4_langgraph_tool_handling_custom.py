"""
Task 4: Tool handling with LangChain tool calling.

Includes:
- calculator tool from Task 3
- letter counting tool
- third custom tool (text_stats)
- dynamic dispatch via tool_map
"""

import argparse
import ast
import json
import math
import os
from typing import Any

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI


def area_circle(r: float) -> float:
    return math.pi * r * r


def circumference_circle(r: float) -> float:
    return 2 * math.pi * r


def area_rectangle(width: float, height: float) -> float:
    return width * height


def area_triangle(base: float, height: float) -> float:
    return 0.5 * base * height


def volume_sphere(r: float) -> float:
    return (4.0 / 3.0) * math.pi * r**3


def volume_cylinder(r: float, h: float) -> float:
    return math.pi * r * r * h


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
    "volume_sphere": volume_sphere,
    "volume_cylinder": volume_cylinder,
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


@tool
def calculator(expression: str, precision: int = 6) -> str:
    """Evaluate a math expression with trig and geometry helper functions."""
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
    count = haystack.count(needle)
    return json.dumps(
        {
            "ok": True,
            "text": text,
            "letter": letter,
            "case_sensitive": case_sensitive,
            "count": count,
        }
    )


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


def run_agent(user_query: str, *, model: str, max_iterations: int) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    tools = [get_weather, calculator, count_letter, text_stats]
    tool_map = {t.name: t for t in tools}
    llm = ChatOpenAI(model=model, temperature=0).bind_tools(tools)

    messages: list[Any] = [
        SystemMessage(
            content=(
                "You are a helpful assistant. Use tools for weather, letter counting, text stats, "
                "and math. Prefer tool use for counting/math instead of mental arithmetic."
            )
        ),
        HumanMessage(content=user_query),
    ]

    print(f"\nUser: {user_query}\n")
    for iteration in range(max_iterations):
        print(f"--- Iteration {iteration + 1} ---")
        response = llm.invoke(messages)

        if response.tool_calls:
            print(f"LLM requested {len(response.tool_calls)} tool call(s).")
            messages.append(response)
            for tool_call in response.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]
                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")

                if function_name in tool_map:
                    result = tool_map[function_name].invoke(function_args)
                else:
                    result = json.dumps({"ok": False, "error": f"Unknown function {function_name}"})

                print(f"  Result: {result}")
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            print()
            continue

        final_text = response.content if isinstance(response.content, str) else str(response.content)
        print(f"Assistant: {final_text}\n")
        return final_text

    return "Max iterations reached."


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4 tool handling with LangChain.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument(
        "--query",
        action="append",
        help="Repeat --query for multiple runs. If omitted, built-in test prompts are used.",
    )
    args = parser.parse_args()

    queries = args.query or [
        "How many s are in Mississippi riverboats?",
        "Are there more i's than s's in Mississippi riverboats? Give the difference too.",
        (
            "What is the sin of the difference between the number of i's and s's in "
            "Mississippi riverboats?"
        ),
        "What is the weather in Tokyo, and how many unique letters are in that city name?",
    ]

    for i, query in enumerate(queries, start=1):
        print("=" * 68)
        print(f"TEST {i}")
        print("=" * 68)
        run_agent(query, model=args.model, max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
