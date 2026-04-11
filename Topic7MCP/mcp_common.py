from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from requests import Response
from dotenv import load_dotenv

load_dotenv()

DEFAULT_ASTA_ENDPOINT = "https://asta-tools.allen.ai/mcp/v1"


class MCPClient:
    def __init__(
        self,
        endpoint: str = DEFAULT_ASTA_ENDPOINT,
        api_key: str | None = None,
        timeout_sec: int = 60,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key or os.getenv("ASTA_API_KEY")
        self.timeout_sec = timeout_sec

        if not self.api_key:
            raise RuntimeError(
                "ASTA_API_KEY is not set. Export it first (or place it in a local .env file)."
            )

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

    def rpc(self, method: str, params: dict[str, Any] | None = None, request_id: int = 1) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        resp = requests.post(
            self.endpoint,
            headers=self._headers,
            json=payload,
            timeout=self.timeout_sec,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            body = resp.text.strip()
            if body:
                raise RuntimeError(f"MCP HTTP error for method '{method}': {body}") from exc
            raise
        data = _decode_rpc_response(resp)
        if "error" in data:
            raise RuntimeError(f"MCP error for method '{method}': {data['error']}")
        return data

    def list_tools(self) -> list[dict[str, Any]]:
        data = self.rpc("tools/list", {}, request_id=1)
        return data.get("result", {}).get("tools", [])

    def call_tool_raw(self, name: str, arguments: dict[str, Any], request_id: int = 2) -> dict[str, Any]:
        return self.rpc(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
            request_id=request_id,
        )

    def call_tool_text(self, name: str, arguments: dict[str, Any], request_id: int = 2) -> str:
        raw = self.call_tool_raw(name=name, arguments=arguments, request_id=request_id)
        return extract_mcp_text(raw)


def extract_mcp_text(raw_response: dict[str, Any]) -> str:
    """Flatten MCP `result.content` payload into a single text string."""
    result = raw_response.get("result", {})
    structured_content = result.get("structuredContent")

    if isinstance(structured_content, str):
        return structured_content

    if isinstance(structured_content, (dict, list)):
        return json.dumps(structured_content, ensure_ascii=True)

    content = result.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if not isinstance(item, dict):
                parts.append(str(item))
                continue

            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue

            nested = item.get("content")
            if isinstance(nested, str):
                parts.append(nested)
                continue

            parts.append(json.dumps(item, ensure_ascii=True))

        flattened = "\n".join(p for p in parts if p)
        return flattened.strip()

    if "text" in result and isinstance(result["text"], str):
        return result["text"]

    return json.dumps(raw_response, ensure_ascii=True)


def _decode_rpc_response(resp: Response) -> dict[str, Any]:
    content_type = resp.headers.get("Content-Type", "").lower()

    if "text/event-stream" in content_type:
        return _parse_sse_jsonrpc(resp.text)

    return resp.json()


def _parse_sse_jsonrpc(body: str) -> dict[str, Any]:
    current_event = "message"
    data_lines: list[str] = []

    for raw_line in body.splitlines():
        line = raw_line.rstrip("\r")

        if not line:
            parsed = _finalize_sse_event(current_event, data_lines)
            if parsed is not None:
                return parsed
            current_event = "message"
            data_lines = []
            continue

        if line.startswith(":"):
            continue

        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip() or "message"
            continue

        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())

    parsed = _finalize_sse_event(current_event, data_lines)
    if parsed is not None:
        return parsed

    raise RuntimeError("MCP server returned SSE data without a JSON-RPC message")


def _finalize_sse_event(event_name: str, data_lines: list[str]) -> dict[str, Any] | None:
    if event_name != "message" or not data_lines:
        return None

    payload = "\n".join(data_lines).strip()
    if not payload:
        return None

    parsed = json.loads(payload)
    if isinstance(parsed, dict):
        return parsed
    raise RuntimeError(f"Unexpected SSE payload type: {type(parsed).__name__}")


def parse_json_maybe(text: str) -> Any:
    """Parse JSON if possible, even when wrapped in markdown fences."""
    if text is None:
        return None

    stripped = text.strip()
    if not stripped:
        return None

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = stripped.find(open_char)
        end = stripped.rfind(close_char)
        if start >= 0 and end > start:
            candidate = stripped[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    return None


def tool_name_or_none(tools: list[dict[str, Any]], preferred_names: list[str]) -> str | None:
    available = {t.get("name"): t for t in tools if t.get("name")}

    for preferred in preferred_names:
        if preferred in available:
            return preferred

    preferred_lower = [p.lower() for p in preferred_names]
    for name in available:
        name_lower = name.lower()
        if any(p in name_lower for p in preferred_lower):
            return name

    return None


def mcp_to_openai_tool(mcp_tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.get("name", "unknown_tool"),
            "description": mcp_tool.get("description", ""),
            "parameters": mcp_tool.get("inputSchema", {"type": "object", "properties": {}}),
        },
    }


def _find_first_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        prioritized_keys = [
            "data",
            "papers",
            "results",
            "items",
            "citations",
            "references",
            "authors",
        ]

        for key in prioritized_keys:
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

        for value in data.values():
            found = _find_first_list(value)
            if found:
                return found

    return []


def normalize_rows_from_tool_text(tool_text: str) -> list[dict[str, Any]]:
    parsed = parse_json_maybe(tool_text)
    if parsed is None:
        return []
    return _find_first_list(parsed)


def unwrap_embedded_paper(record: dict[str, Any]) -> dict[str, Any]:
    """Some tools nest paper metadata under keys like `paper` or `citingPaper`."""
    nested_keys = ["paper", "citingPaper", "citedPaper", "reference", "citation"]
    for key in nested_keys:
        value = record.get(key)
        if isinstance(value, dict):
            merged = dict(value)
            for keep in ("year", "citationCount", "influentialCitationCount"):
                if keep in record and keep not in merged:
                    merged[keep] = record[keep]
            return merged
    return record
