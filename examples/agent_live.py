"""Kiểm thử chạy thật Agent với LLM (Groq API) và chấm điểm tự động bằng Verifier."""
import asyncio
import json
import sys
from pathlib import Path

# Đảm bảo UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from finco.agent import Agent
from eval.sandbox_db import isolated_db_sandbox
from eval.verifier import verify_decision_record, ExpectedDecision


async def run_single_application(app_id: str = "APP-003"):
    print("\n" + "=" * 70)
    print(f" BẮT ĐẦU CHẠY THẬT AGENT TRÊN HỒ SƠ: {app_id}")
    print("=" * 70)

    # 1. Đọc kỳ vọng chuẩn từ tests/expected_decisions.json
    expected_path = REPO_ROOT / "tests" / "expected_decisions.json"
    with open(expected_path, "r", encoding="utf-8") as f:
        expected_list = json.load(f)
    expected_data = next((item for item in expected_list if item["app_id"] == app_id), None)
    
    if not expected_data:
        print(f"[-] Không tìm thấy dữ liệu kỳ vọng cho {app_id}")
        return

    expected = ExpectedDecision.from_dict(expected_data)

    # 2. Chạy Agent trong môi trường Sandbox DB cô lập
    with isolated_db_sandbox() as db_path:
        print(f"[Sandbox] Đang chạy với DB tạm: {db_path.name}")
        
        agent = Agent(model="openai/gpt-oss-20b", verbose=True)
        res = await agent.run(app_id)

        print("\n" + "=" * 70)
        print(" KẾT QUẢ LƯỢT CHẠY CỦA AGENT")
        print("=" * 70)
        print(f"Trạng thái: {'THÀNH CÔNG' if res.success else 'THẤT BẠI'}")
        print(f"Số bước đã đi: {len(res.trace)}")
        print(f"Số tool đã gọi: {len(res.trace)} ({' -> '.join(res.tool_calls)})")
        if res.usage:
            total = res.usage.prompt_tokens + res.usage.completion_tokens
            print(f"Token tiêu thụ: Calls={res.usage.calls}, Prompt={res.usage.prompt_tokens}, Completion={res.usage.completion_tokens}, Total={total}")

        # 3. Chấm điểm bằng Verifier
        submission = res.to_submission()
        verif_res = verify_decision_record(submission, expected)

        print("\n" + "=" * 70)
        print(" BẢNG ĐIỂM VERIFIER (ĐỊNH LƯỢNG)")
        print("=" * 70)
        print(f"1. Khớp Quyết định (Decision) : {'PASS' if verif_res.decision_correct else 'FAIL'}")
        print(f"   -> Agent: {submission.decision} | Kỳ vọng: {expected.decision}")
        print(f"2. Mã nguyên nhân (Root cause): {'PASS' if verif_res.root_cause_correct else 'FAIL'}")
        print(f"   -> Agent: {submission.reason_codes} | Kỳ vọng: {expected.reasons}")
        print(f"3. Trích dẫn tài liệu (Citations): {'PASS' if verif_res.citation_valid else 'FAIL'} (Recall={verif_res.citation_recall:.2f}, Precision={verif_res.citation_precision:.2f})")
        print(f"   -> Agent: {submission.cited_docs}")
        print(f"4. Ghi nhận DB (Read-Your-Writes): {'PASS' if verif_res.read_after_write else 'FAIL'}")
        print(f"5. Quy trình gọi tool (Trace):    {'PASS' if verif_res.trace_valid else 'FAIL'}")
        print(f"6. Lãi suất sử dụng (Rate):       {'PASS' if verif_res.rate_correct else 'FAIL'}")
        print(f"7. Sai số DTI (Math):             {'PASS' if verif_res.dti_correct else 'FAIL'}")
        print(f"   -> Agent DTI: {verif_res.actual_dti} | Kỳ vọng: {expected.dti}")
        print("-" * 70)
        print(f"==> KẾT QUẢ CHUNG: {'ĐỖ TOÀN DIỆN (OVERALL PASS)' if verif_res.is_overall_pass else 'CHƯA ĐẠT (FAILED)'}")
        if not verif_res.is_overall_pass:
            print(f"Lý do: {verif_res.error_message}")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_single_application("APP-003"))
