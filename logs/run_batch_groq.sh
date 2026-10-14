#!/usr/bin/env bash
# Chạy một đợt câu hỏi/hồ sơ bằng Groq (openai/gpt-oss-20b), ghi tiếp vào một log duy nhất.
# Dùng:  bash logs/run_batch_groq.sh A01 A02 C01
# Cuối cùng chấm gộp:  bash logs/run_batch_groq.sh --score
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
unset GROQ_BASE_URL OPENAI_BASE_URL LLM_MODEL   # dùng mặc định: Groq + gpt-oss-20b, khoá GROQ_API_KEY trong .env

# Dùng Python của .venv (khi gọi `bash` từ PowerShell, bash có thể là WSL với python hệ thống thiếu thư viện).
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=python; fi

LOG=logs/eval_gptoss_final.jsonl
OUT=eval/out/gptoss_final
[ -f "$LOG" ] || : > "$LOG"

if [ "$1" = "--score" ]; then
  "$PY" eval/run_eval.py --apps --score-only "$LOG" --out-dir "$OUT"
else
  "$PY" eval/run_eval.py --apps --ids "$@" --resume "$LOG" --pace 8 --out-dir "$OUT"
fi
