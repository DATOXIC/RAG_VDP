from typing import Optional
from mcp.server.fastmcp import FastMCP

from finco import tools

# Khởi tạo MCP Server với tên "finco"
mcp = FastMCP("finco")


@mcp.tool()
def search_policy(query: str, as_of_date: Optional[str] = None, top_k: int = 5) -> str:
    """Tra cứu chính sách tín dụng nội bộ FinCo (RAG hybrid BM25 + Dense).
    Args:
        query: Câu hỏi hoặc từ khóa cần tra cứu quy định chính sách.
        as_of_date: Ngày hiệu lực cần lọc chính sách (YYYY-MM-DD), thường là ngày nộp hồ sơ.
        top_k: Số lượng đoạn trích tối đa cần lấy (mặc định 5).
    """
    return tools.search_policy(query=query, as_of_date=as_of_date, top_k=top_k)


@mcp.tool()
def get_application(app_id: str) -> str:
    """Lấy thông tin chi tiết một hồ sơ vay từ cơ sở dữ liệu.
    Args:
        app_id: Mã hồ sơ vay (ví dụ: 'APP-001', 'APP-015').
    """
    return tools.get_application(app_id=app_id)


@mcp.tool()
def calc_dti(app_id: str, annual_rate: str) -> str:
    """Tính toán tỷ lệ nợ trên thu nhập (DTI) và khoản trả nợ hàng tháng (PMT) bằng code.
    Args:
        app_id: Mã hồ sơ vay (ví dụ: 'APP-001').
        annual_rate: Lãi suất năm tra cứu được từ chính sách (ví dụ: '0.155' hoặc '15.5%').
    """
    return tools.calc_dti(app_id=app_id, annual_rate=annual_rate)


@mcp.tool()
def submit_decision(
    app_id: str,
    decision: str,
    reason: str,
    cited_docs: list[str],
    reason_codes: list[str],
) -> str:
    """Ghi nhận quyết định thẩm định chính thức vào cơ sở dữ liệu.
    Args:
        app_id: Mã hồ sơ vay (ví dụ: 'APP-001').
        decision: Trạng thái quyết định: 'approve' | 'reject' | 'escalate' | 'need_info'.
        reason: Lời giải thích chi tiết nêu rõ căn cứ và số liệu thẩm định.
        cited_docs: Danh sách tài liệu chính sách làm căn cứ (ví dụ: ['CP-01 v2', 'CP-03']).
        reason_codes: Danh sách mã lý do có cấu trúc, phải khớp decision:
            - approve   -> ['APPROVED']
            - reject    -> AGE, OVERDUE, AMOUNT_RANGE, TERM_RANGE, DTI_EXCEEDED
            - need_info -> MISSING_DOCS, MISSING_DEBT_DECLARATION
            - escalate  -> ESCALATE_MANAGER, ESCALATE_COUNCIL, ESCALATE_EMPLOYEE
    """
    return tools.submit_decision(
        app_id=app_id,
        decision=decision,
        reason=reason,
        cited_docs=cited_docs,
        reason_codes=reason_codes,
    )

@mcp.tool()
def get_decision(app_id: str) -> str:
    """Đọc lại quyết định đã lưu trong cơ sở dữ liệu (phục vụ bước kiểm chứng Read-Your-Writes).
    Args:
        app_id: Mã hồ sơ vay cần kiểm tra (ví dụ: 'APP-001').
    """
    return tools.get_decision(app_id=app_id)

def main() -> None:
    """Khởi động MCP Server qua stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
