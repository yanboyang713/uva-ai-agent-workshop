"""
Task 3: Manual tool handling with OpenAI Chat Completions.

Implements:
- weather tool
- custom calculator tool with geometric helper functions
- json.loads for tool input parsing
- json.dumps for tool output formatting
"""

import argparse
import ast
import json
import math
import os
from typing import Any

from openai import OpenAI


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
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
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


def weather_tool(payload_json: str) -> str:
    args = json.loads(payload_json)
    location = str(args.get("location", "")).strip()
    weather_data = {
        "san francisco": "Sunny, 72F",
        "new york": "Cloudy, 55F",
        "london": "Rainy, 48F",
        "tokyo": "Clear, 65F",
        "charlottesville": "Partly cloudy, 68F",
    }
    forecast = weather_data.get(location.lower(), f"Weather data not available for {location}")
    return json.dumps({"location": location, "forecast": forecast})


def calculator_tool(payload_json: str) -> str:
    args = json.loads(payload_json)
    expression = str(args.get("expression", "")).strip()
    precision = int(args.get("precision", 6))
    if not expression:
        return json.dumps({"ok": False, "error": "Missing expression"})

    try:
        value = safe_eval(expression)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "expression": expression, "error": str(exc)})

    return json.dumps(
        {
            "ok": True,
            "expression": expression,
            "result": round(value, precision),
            "precision": precision,
        }
    )


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name."},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a math expression with trigonometric and geometry helpers. "
                "Available helpers: sin, cos, tan, sqrt, area_circle, circumference_circle, "
                "area_rectangle, area_triangle, volume_sphere, volume_cylinder."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "precision": {"type": "integer", "default": 6},
                },
                "required": ["expression"],
            },
        },
    },
]


def run_agent(
    user_query: str,
    *,
    model: str,
    max_iterations: int,
    force_tool: str | None = None,
) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    tool_map = {
        "get_weather": weather_tool,
        "calculator": calculator_tool,
    }

    messages: list[Any] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use tools for weather and calculations. "
                "Do not fabricate weather. Use calculator for numeric answers."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    tool_choice: Any = "auto"
    if force_tool:
        tool_choice = {"type": "function", "function": {"name": force_tool}}

    print(f"\nUser: {user_query}\n")
    for iteration in range(max_iterations):
        print(f"--- Iteration {iteration + 1} ---")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice=tool_choice,
        )

        assistant_message = response.choices[0].message
        if assistant_message.tool_calls:
            print(f"LLM requested {len(assistant_message.tool_calls)} tool call(s).")
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                payload_json = json.dumps(function_args)

                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")

                if function_name in tool_map:
                    result = tool_map[function_name](payload_json)
                else:
                    result = json.dumps({"ok": False, "error": f"Unknown function {function_name}"})

                print(f"  Result: {result}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": result,
                    }
                )
            print()
            continue

        final_text = assistant_message.content or ""
        print(f"Assistant: {final_text}\n")
        return final_text

    return "Max iterations reached."


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3 manual tool-use agent.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--force-tool", choices=["get_weather", "calculator"])
    parser.add_argument(
        "--query",
        action="append",
        help="Repeat --query for multiple runs. If omitted, built-in test prompts are used.",
    )
    args = parser.parse_args()

    queries = args.query or [
        "What's the weather in Charlottesville?",
        "What is area_circle(3) and circumference_circle(3)?",
        "What is sin(area_rectangle(3, 4) - area_triangle(6, 4))?",
    ]

    for i, query in enumerate(queries, start=1):
        print("=" * 68)
        print(f"TEST {i}")
        print("=" * 68)
        run_agent(
            query,
            model=args.model,
            max_iterations=args.max_iterations,
            force_tool=args.force_tool,
        )


if __name__ == "__main__":
    main()
