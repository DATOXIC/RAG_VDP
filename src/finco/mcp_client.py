"""MCP Client cho FinCo (Bám sát bài học Tuần 7 - Week 7).

Cung cấp:
1. Client: Kết nối MCP Server qua stdio (thư viện chính thức mcp).
2. get_default_server_params: Cấu hình kết nối tới server finco.
3. to_openai_tools: Chuyển đổi danh sách tools sang format Groq/OpenAI.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_default_server_params() -> StdioServerParameters:
    """Tham số kết nối tới FinCo MCP Server."""
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}"
    # Windows: stderr của server mặc định là cp1252 nên log tiếng Việt thành "đ...".
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("FINCO_LOG_LEVEL", "WARNING")
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "finco.mcp_server"],
        env=env,
    )


def to_openai_tools(mcp_tools: list[Any], require_thought: bool = True) -> list[dict[str, Any]]:
    """ADAPTER: Chuyển đổi MCP Tools sang định dạng Tools của Groq/OpenAI, 
    tự động tiêm trường 'thought' lên đầu mỗi tool để ép LLM suy nghĩ trước khi gọi."""
    openai_tools = []
    
    for tool in mcp_tools:
        # 1. Lấy schema gốc từ MCP
        schema = dict(getattr(tool, "input_schema", getattr(tool, "inputSchema", {})))
        properties = dict(schema.get("properties", {}))
        required = list(schema.get("required", []))

        # 2. Tiêm trường 'thought' vào vị trí đầu tiên
        if require_thought:
            new_properties = {
                "thought": {
                    "type": "string",
                    "description": (
                        "Suy nghĩ ngắn gọn (1-2 câu): Tại sao gọi công cụ này? "
                        "Cần dữ liệu gì và dự định làm gì tiếp theo với kết quả?"
                    ),
                }
            }
            # Ghép các tham số nghiệp vụ gốc vào sau thought
            new_properties.update(properties)
            properties = new_properties

            # Đánh dấu thought là bắt buộc
            if "thought" not in required:
                required.insert(0, "thought")

        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
        
    return openai_tools


async def main():
    """Demo kết nối và gọi tool chuẩn theo giáo trình Tuần 7."""
    server = get_default_server_params()

    async with Client(server) as client:
        # 1. DISCOVERY: Lấy danh sách tools (.tools)
        response = await client.list_tools()
        tools = response.tools
        print("Danh sách công cụ từ MCP Server:")
        for t in tools:
            print(f"  + [{t.name}]: {t.description.splitlines()[0]}")

        # 2. ADAPTER: Chuyển đổi sang format Groq / OpenAI
        groq_tools = to_openai_tools(tools)
        print(f"\n-> Đã chuyển đổi thành công {len(groq_tools)} tools sang định dạng LLM.")

        # 3. GỌI THỬ TOOL
        mcp_result = await client.call_tool("calc_dti", {"app_id": "APP-003", "annual_rate": "0.155"})
        print(f"\nKết quả thực hiện tool calc_dti:")
        print(mcp_result.content[0].text)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())