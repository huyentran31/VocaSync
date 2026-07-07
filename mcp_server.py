"""
mcp_server.py — thin MCP (Model Context Protocol) server over stdio (Day-2).

Exposes the registered tools (currently 10, incl. conceptnet_lookup + stage_for_review)
from `agent/registry.py` to any MCP client via JSON-RPC 2.0 on stdin/stdout. The list is
read dynamically from the registry, so a new tool appears here automatically. The
Python functions remain the SINGLE SOURCE OF TRUTH; this is a thin wrapper with NO
business logic (AGENTS.md: MCP = access, not know-how).

No external `mcp` SDK dependency — a minimal self-contained stdin/stdout loop, so it
runs anywhere (stdio is the simplest transport, Day-2).

Methods:
  initialize                      -> protocol version + capabilities + serverInfo
  notifications/initialized       -> (notification, ignored)
  tools/list                      -> [{name, description, inputSchema}]   (schema from registry)
  tools/call {name, arguments}    -> tool result as text content (tool errors -> isError, never crash)

Run:   python mcp_server.py
Test:  printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python mcp_server.py
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "agent"), os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "legacy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from registry import TOOLS, call_tool  # single source of truth for the 10 tools

SERVER_INFO = {"name": "vocabgraph-agent", "version": "1.0.0"}
PROTOCOL_VERSION = "2024-11-05"


def _tools_list() -> list[dict]:
    return [
        {"name": name, "description": t["description"], "inputSchema": t["schema"]}
        for name, t in TOOLS.items()
    ]


def _handle(req: dict):
    """Return a JSON-RPC response dict, or None for notifications (no id / no reply)."""
    rid = req.get("id")
    method = req.get("method")
    params = req.get("params") or {}

    def ok(result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def err(code, msg):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}

    if method == "initialize":
        return ok({"protocolVersion": PROTOCOL_VERSION,
                   "capabilities": {"tools": {}},
                   "serverInfo": SERVER_INFO})
    if method in ("notifications/initialized", "initialized"):
        return None  # notification — no response
    if method == "tools/list":
        return ok({"tools": _tools_list()})
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name not in TOOLS:
            return err(-32602, f"unknown tool: {name}")
        try:
            result = call_tool(name, args)
            text = json.dumps(result, ensure_ascii=False, default=str)
            return ok({"content": [{"type": "text", "text": text}]})
        except Exception as e:
            # tool/system error -> report as MCP isError content; keep the server alive
            return ok({"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
    if rid is None:
        return None  # unknown notification
    return err(-32601, f"method not found: {method}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
            ) + "\n")
            sys.stdout.flush()
            continue
        resp = _handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
