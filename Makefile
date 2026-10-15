# FinCo - Trợ lý thẩm định hồ sơ vay.  Chạy:  make setup && make demo
# Máy không có make: dùng trực tiếp `python run.py setup`, `python run.py demo`, `python run.py eval`.

PYTHON ?= python
VENV   := .venv
ifeq ($(OS),Windows_NT)
  PY := $(VENV)/Scripts/python.exe
else
  PY := $(VENV)/bin/python
endif

.PHONY: setup demo run eval eval-score test retrieval-eval interop clean

# Tạo venv, cài thư viện, tạo DB hồ sơ, tải mô hình embedding (lần đầu ~2 GB), kiểm tra khóa API
setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) run.py setup

# Demo không tương tác: thẩm định APP-014 và APP-015 (cùng số liệu, khác phiên bản chính sách)
# + một câu hỏi ngoài tài liệu. Đổi hồ sơ:  make demo APPS="APP-003 APP-018"
demo:
	$(PY) run.py demo $(APPS)

# REPL tương tác
run:
	$(PY) run.py

# Chạy 16 câu trong tests/testset.jsonl. Ghi logs/eval_<thời gian>.jsonl và eval/out/*.
# Thêm tham số:  make eval ARGS="--apps --pace 8"   hoặc chỉ vài hồ sơ:  ARGS="--ids APP-001 APP-003"
eval:
	$(PY) eval/run_eval.py $(ARGS)

# Chấm lại một log đã có, KHÔNG gọi LLM:  make eval-score LOG=logs/eval_XXXX.jsonl
eval-score:
	$(PY) eval/run_eval.py --score-only $(LOG)

# Đo riêng tầng truy xuất (không cần khóa API)
retrieval-eval:
	$(PY) eval/retrieval_eval.py

# Host thứ hai: client JSON-RPC thuần (không SDK, không LLM) gọi MCP server FinCo trên bản sao DB
interop:
	$(PY) examples/client2.py

# Toàn bộ test offline: agent giả, chấm điểm, MCP, truy xuất. Không gọi LLM.
test:
	$(PY) -m unittest discover -s tests -p "test_*.py"

clean:
	$(PYTHON) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
