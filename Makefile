# FinCo - Trợ lý thẩm định hồ sơ vay.  Chạy:  make setup && make demo
# Máy không có make: dùng trực tiếp `python run.py setup`, `python run.py demo`, `python run.py eval`.

PYTHON ?= python
VENV   := .venv
ifeq ($(OS),Windows_NT)
  PY := $(VENV)/Scripts/python.exe
else
  PY := $(VENV)/bin/python
endif

.PHONY: all setup demo run eval eval-quick eval-score ragas test retrieval-eval interop check clean

# Lệnh duy nhất tái lập demo từ máy sạch:  make all   (= setup + demo)
# Tái lập kết quả đánh giá:               make eval  (37 ca: 16 câu hỏi + 21 hồ sơ)
all: setup demo

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

# Mặc định chạy ĐỦ 37 ca (16 câu trong tests/testset.jsonl + 21 hồ sơ). Ghi logs/eval_<thời gian>.jsonl và eval/out/*.
# Đổi tham số:  make eval ARGS="--apps --pace 8"   hoặc chỉ vài hồ sơ:  ARGS="--apps --ids APP-001 APP-003"
# Chỉ 16 câu hỏi (không hồ sơ):  make eval ARGS=""
ARGS ?= --apps
eval:
	$(PY) eval/run_eval.py $(ARGS)

# Bộ nhanh: 6 ca đại diện (T01, V01, N01, APP-001, APP-014, APP-003), ~150k token, ghi eval/out/quick.
eval-quick:
	$(PY) run.py eval-quick $(ARGS_QUICK)

# Chấm lại một log đã có, KHÔNG gọi LLM:  make eval-score LOG=logs/eval_XXXX.jsonl  (mặc định: log cấu hình chính)
eval-score:
	$(PY) run.py eval-score $(LOG)

# Đánh giá bổ sung bằng RAGAS từ log có sẵn (không chạy lại Actor; cần GEMINI_API_KEY và requirements-eval.txt).
# Ghi eval/out/ragas/. Dừng giữa chừng thì chạy lại, tự tiếp tục từ checkpoint.
ragas:
	$(PY) -m pip install -r requirements-eval.txt
	$(PY) eval/ragas_eval.py $(RAGAS_ARGS)

# Đo riêng tầng truy xuất (không cần khóa API)
retrieval-eval:
	$(PY) eval/retrieval_eval.py

# Host thứ hai: client JSON-RPC thuần (không SDK, không LLM) gọi MCP server FinCo trên bản sao DB
interop:
	$(PY) examples/client2.py

# Kiểm chứng không cần khóa API và không gọi LLM: test offline, đo truy xuất, host thứ hai, chấm lại log chính.
CHECK_LOG ?= logs/eval_20261014_202454.jsonl
check: test retrieval-eval interop
	$(PY) eval/run_eval.py --score-only $(CHECK_LOG) --apps --out-dir eval/out/check

# Toàn bộ test offline: agent giả, chấm điểm, MCP, truy xuất. Không gọi LLM.
test:
	$(PY) -m unittest discover -s tests -p "test_*.py"

clean:
	$(PYTHON) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
