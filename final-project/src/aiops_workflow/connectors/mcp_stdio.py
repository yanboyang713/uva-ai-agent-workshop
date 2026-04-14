from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _coerce_text_payload(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return {"text": stripped}


@dataclass
class StdioMCPClient:
    command: str
    args: list[str]
    env: dict[str, str] | None = None

    async def _call_tool_async(self, name: str, arguments: dict[str, Any]) -> Any:
        params = StdioServerParameters(command=self.command, args=self.args, env=self.env or {})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments=arguments)
                if getattr(result, "structuredContent", None) is not None:
                    return result.structuredContent

                texts: list[str] = []
                for block in result.content:
                    if hasattr(block, "text"):
                        texts.append(block.text)
                return _coerce_text_payload("\n".join(texts))

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return asyncio.run(self._call_tool_async(name, arguments))
