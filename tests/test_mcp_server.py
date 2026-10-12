"""Kiểm thử tự động cho MCP Server và MCP Client (Tuần 7).

Kiểm tra:
1. Server khởi động và bắt tay giao thức MCP qua stdio.
2. Server phơi đúng 5 tools với schema đầy đủ.
3. Server thực thi thành công một lệnh gọi tool (calc_dti).
4. Client (chuẩn mcp v2 của lớp) hoạt động mượt mà trong khối `async with Client(server) as client:`.
5. to_openai_tools() sinh đúng 5 schema chuẩn OpenAI/Groq.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from finco.mcp_client import Client, get_default_server_params, to_openai_tools


class TestMCPServerAndClient(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_client_workflow(self):
        """Kiểm tra toàn bộ luồng Client và Server theo đúng giáo trình Tuần 7."""
        server = get_default_server_params()

        async with Client(server) as client:
            # 1. DISCOVERY: Lấy danh sách tools
            response = await client.list_tools()
            tools = response.tools
            tool_names = [tool.name for tool in tools]

            expected_tools = [
                "search_policy",
                "get_application",
                "calc_dti",
                "submit_decision",
                "get_decision",
            ]
            for exp in expected_tools:
                self.assertIn(exp, tool_names, f"Thiếu tool {exp} trong MCP Server")

            # 2. ADAPTER: Chuyển sang định dạng Groq / OpenAI
            openai_schemas = to_openai_tools(tools)
            self.assertEqual(len(openai_schemas), 5)
            for s in openai_schemas:
                self.assertEqual(s["type"], "function")
                self.assertIn("name", s["function"])
                self.assertIn("parameters", s["function"])

            # 3. GỌI THỰC THI TOOL
            call_result = await client.call_tool(
                name="calc_dti",
                arguments={"app_id": "APP-003", "annual_rate": "0.155"},
            )

            self.assertIsNotNone(call_result.content)
            self.assertTrue(len(call_result.content) > 0)
            text_out = call_result.content[0].text
            self.assertIn("APP-003", text_out)
            self.assertIn("62.43%", text_out)


if __name__ == "__main__":
    unittest.main()
