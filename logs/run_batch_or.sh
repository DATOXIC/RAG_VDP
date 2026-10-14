#!/usr/bin/env bash
# Chạy một đợt hồ sơ trên OpenRouter, ghi tiếp vào logs/eval_20261014_170836.jsonl.
# Dùng:  bash logs/run_batch_or.sh APP-003 APP-004 APP-005
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
KEY=$(python -c "from dotenv import dotenv_values;print(dotenv_values('.env').get('OPENROUTER_API_KEY',''))")
export GROQ_API_KEY="$KEY"
export GROQ_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="nvidia/nemotron-3-super-120b-a12b:free"
python eval/run_eval.py --apps --ids "$@" --resume logs/eval_20261014_170836.jsonl --pace 5 --out-dir eval/out/openrouter
