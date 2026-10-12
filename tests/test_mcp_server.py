"""Kiểm thử tự động cho MCP Server và MCP Client.

Kiểm tra:
1. Server khởi động và bắt tay giao thức MCP qua stdio.
2. Server phơi đúng 5 tools với schema đầy đủ.
3. Server thực thi thành công một lệnh gọi tool (calc_dti).
4. MCPClient (Sync Wrapper) hoạt động mượt mà trong khối `with MCPClient():`
5. MCPClient.get_openai_tools() sinh đúng 5 schema chuẩn OpenAI/Groq.
"""

import os
import sys
import unittest
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from finco.mcp_client import MCPClient


class TestMCPServer(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_server_raw_stdio(self):
        """Kiểm tra server cấp thấp qua stdio client trực tiếp."""
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}"

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "finco.mcp_server"],
            env=env,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_response = await session.list_tools()
                tool_names = [tool.name for tool in tools_response.tools]

                expected_tools = [
                    "search_policy",
                    "get_application",
                    "calc_dti",
                    "submit_decision",
                    "get_decision",
                ]
                for exp in expected_tools:
                    self.assertIn(exp, tool_names, f"Thiếu tool {exp} trong MCP Server")

                call_result = await session.call_tool(
                    name="calc_dti",
                    arguments={"app_id": "APP-003", "annual_rate": "0.155"},
                )

                self.assertIsNotNone(call_result.content)
                self.assertTrue(len(call_result.content) > 0)
                text_out = call_result.content[0].text
                self.assertIn("APP-003", text_out)
                self.assertIn("DTI", text_out)


class TestMCPClientSync(unittest.TestCase):
    def test_sync_client_workflow(self):
        """Kiểm tra toàn bộ luồng của MCPClient ở chế độ đồng bộ cho Agent."""
        with MCPClient() as client:
            # 1. Khám phá tools
            tools = client.list_tools()
            self.assertEqual(len(tools), 5)
            names = [t.name for t in tools]
            self.assertIn("search_policy", names)
            self.assertIn("calc_dti", names)

            # 2. Sinh schema OpenAI/Groq
            openai_schemas = client.get_openai_tools()
            self.assertEqual(len(openai_schemas), 5)
            for s in openai_schemas:
                self.assertEqual(s["type"], "function")
                self.assertIn("name", s["function"])
                self.assertIn("parameters", s["function"])

            # 3. Gọi công cụ
            res = client.call_tool("calc_dti", {"app_id": "APP-003", "annual_rate": "0.155"})
            self.assertIn("APP-003", res)
            self.assertIn("62.43%", res)


if __name__ == "__main__":
    unittest.main()
