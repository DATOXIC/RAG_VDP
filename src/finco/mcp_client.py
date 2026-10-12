import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parents[2] 


def get_default_server_params() -> StdioServerParameters:
    """Cấu hình tham số để khởi động FinCo MCP Server."""
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}"
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "finco.mcp_server"],
        env=env,
    ) 


class Client:
    """MCP Client chuẩn async, cách dùng giống hệt bài học Tuần 7."""

    def __init__(self, server_params: StdioServerParameters = None):
        self.server_params = server_params or get_default_server_params()
        self._stack = AsyncExitStack()
        self.session: ClientSession = None

    async def __aenter__(self):
        # 1. Kết nối tiến trình qua stdio
        read, write = await self._stack.enter_async_context(stdio_client(self.server_params))
        
        # 2. Tạo phiên làm việc JSON-RPC và bắt tay
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Đóng kết nối sạch sẽ khi thoát with
        await self._stack.aclose()

    async def list_tools(self):
        """1. DISCOVERY: Lấy danh sách tools từ server."""
        return await self.session.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        """2. EXECUTE: Gọi thực thi tool trên server."""
        return await self.session.call_tool(name=name, arguments=arguments)


def to_openai_tools(mcp_tools) -> list[dict]:
    """ADAPTER: Chuyển đổi MCP Tools sang định dạng của Groq/OpenAI (giống hệt bài học)."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": getattr(tool, "inputSchema", getattr(tool, "input_schema", {})),
            },
        }
        for tool in mcp_tools
    ]