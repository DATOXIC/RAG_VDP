# Chỉ mục kết quả chấm (`eval/out/`)

Mỗi thư mục chứa `per_question.csv`/`agent_eval_results.csv`, `summary.json`, `summary.png` do `eval/run_eval.py` sinh ra khi chấm một log. Mã cấu hình A–J xem `docs/ket-qua-danh-gia.md` và `logs/README.md`. Chấm lại bất kỳ log nào: `make eval-score LOG=logs/<file>.jsonl`.

## Kết quả dùng trong báo cáo

| Thư mục | Cấu hình | Đạt/37 |
|---|---|---|
| `ab_nocritic` | **A**: gpt-oss-120b, không critic (số liệu chính) | 34 |
| `agent_full` | **B**: 120b + critic-agent Gemini | 32 |
| `local_nocritic` | **C**: Qwen2.5-7B, không critic | 10 |
| `local_critic_v2` | **D**: Qwen2.5-7B + critic một lần gọi bản sửa | 9 |
| `qwen` | **E**: Qwen2.5-14B, không critic | 8 |
| `vuong20_nocritic` | **G**: gpt-oss:20b trên Ollama | 23 |
| `g20_none_w1..w4` | **H**: gpt-oss-20b Groq, không critic (4 shard gộp lại: 33) | 33 |
| `g20_single_w1..w4` | **I**: 20b + critic một lần gọi Gemini (4 shard: 32) | 32 |
| `g20_agent_w1..w4` | **J**: 20b + critic-agent Gemini (4 shard: 32) | 32 |
| `g20_comparison.csv` | bảng từng ca của G, H, I, J | |
| `compare/` | đồ thị và bảng so sánh A, H, C do `make compare` sinh ra (`compare.png`, `compare.csv`) | |
| `ragas/` | RAGAS bổ sung trên log của A (12 câu có đáp án): điểm từng câu và checkpoint |
| `retrieval_per_question*.csv` | đo tầng truy xuất, có và không lọc ngày (bonus) | |

## `archive/` (lần chạy đầu và thử nghiệm sớm, giữ để truy vết, không dùng làm số liệu)

| Thư mục trong `archive/` | Là gì |
|---|---|
| `ab_critic` | critic bản đầu trên 120b: 20/37, **đã bị thay** bởi critic đã sửa và critic-agent (B) |
| `local_critic` | critic bản đầu trên Qwen 7B: 8/37, đã thay bằng `local_critic_v2` |
| `local_critic_v2_w1..w3`, `agent_full_w*` | shard hoặc lần chạy lại của D và B |
| `ab_pilot`, `local_pilot`, `agent_smoke*`, `vuong20_smoke`, `openrouter*` | chạy thử 1–3 ca, gồm thử model `nemotron` trên OpenRouter |

## Sinh tự động, không phải số liệu

- `summary.*`, `per_question.csv`, `agent_eval_results.csv` ở gốc: kết quả của lần chấm gần nhất, bị ghi đè mỗi lần chạy `make eval`/`eval-score`.
- `check/`, `quick/`: do `make check` và `make eval-quick` ghi.
