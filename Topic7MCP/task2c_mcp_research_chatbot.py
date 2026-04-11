"""Exercise C (MCP): Asta-powered research chatbot with dynamic tool discovery."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from mcp_common import MCPClient, mcp_to_openai_tool

load_dotenv()

SYSTEM_PROMPT = (
    "You are a research assistant with access to Semantic Scholar tools via MCP. "
    "Use tools when factual retrieval is needed. Keep answers concise and cite paper titles."
)


def _assistant_to_message_dict(assistant_msg: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_msg.content or "",
    }

    if assistant_msg.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in assistant_msg.tool_calls
        ]

    return payload


def _parse_tool_args(raw_args: str) -> dict[str, Any]:
    if not raw_args:
        return {}

    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError:
        return {}

    if isinstance(parsed, dict):
        return parsed

    return {}


def run_turn(
    openai_client: OpenAI,
    mcp_client: MCPClient,
    model: str,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    max_steps: int = 8,
) -> str:
    """Run one full user turn, including chained tool calls."""
    for _ in range(max_steps):
        completion = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )
        assistant_msg = completion.choices[0].message

        if not assistant_msg.tool_calls:
            final_text = assistant_msg.content or ""
            messages.append({"role": "assistant", "content": final_text})
            return final_text

        messages.append(_assistant_to_message_dict(assistant_msg))

        for tool_call in assistant_msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = _parse_tool_args(tool_call.function.arguments)

            print(f"\n[tool] {tool_name}")
            print(f"[args] {json.dumps(tool_args, ensure_ascii=True)}")

            try:
                tool_result = mcp_client.call_tool_text(tool_name, tool_args)
            except Exception as exc:  # nosec - report tool errors to model
                tool_result = f"MCP tool call failed for {tool_name}: {exc}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": tool_result[:12000],
                }
            )

    return "I reached the tool-call step limit before producing a final answer."


def build_tools(client: MCPClient) -> list[dict[str, Any]]:
    mcp_tools = client.list_tools()
    openai_tools = [mcp_to_openai_tool(t) for t in mcp_tools]
    return openai_tools


def interactive_chat(model: str) -> None:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    openai_client = OpenAI(api_key=openai_key)
    mcp_client = MCPClient()

    tools = build_tools(mcp_client)
    print(f"Loaded {len(tools)} MCP tools from Asta.")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    print("Type a question (or 'quit').")
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"quit", "exit", "q"}:
            break
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        answer = run_turn(openai_client, mcp_client, model, tools, messages)
        print(f"\nAssistant: {answer}")


def single_question(model: str, question: str) -> None:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    openai_client = OpenAI(api_key=openai_key)
    mcp_client = MCPClient()

    tools = build_tools(mcp_client)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    answer = run_turn(openai_client, mcp_client, model, tools, messages)
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP Asta research chatbot")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--once", help="Single question mode")
    args = parser.parse_args()

    if args.once:
        single_question(args.model, args.once)
    else:
        interactive_chat(args.model)


if __name__ == "__main__":
    main()
