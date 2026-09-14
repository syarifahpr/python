# PromptBench RAG Evaluation for Flowise

Evaluates a Flowise "prediction" API endpoint (a deployed RAG chatflow) by
sending it a set of labeled question/answer pairs under several prompt
phrasings, and scoring the answers with exact-match / F1 — the same idea
behind [promptbench](https://github.com/microsoft/promptbench)'s prompt
robustness evaluations, applied to a live RAG API instead of a raw LLM.

## How it fits together

- `flowise_client.py` — thin REST client for `POST {url}` with
  `{"question": ...}`, with retries and defensive response parsing.
- `rag_model.py` — wraps the client as a `model(input_text) -> str`
  callable, the same calling convention promptbench's own `LLMModel` uses,
  so promptbench's prompt/templating utilities can be reused unmodified.
- `prompts.py` — default prompt templates (each with a `{question}`
  placeholder) used to probe robustness to phrasing.
- `metrics.py` — SQuAD-style exact-match/F1 scoring. promptbench's own
  `Eval.compute_squad_v2_f1` depends on a resource file
  (`promptbench/metrics/squad_v2/squad_v2.py`) that isn't actually included
  in the published PyPI wheel (0.0.4), so this reimplements the same
  standard formula directly.
- `eval_rag.py` — CLI that ties it together, using promptbench's
  `Prompt` and `InputProcess.basic_format`/`OutputProcess.general` for
  templating/cleanup.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate

# promptbench's dependency `autocorrect==2.6.1` fails to build on modern
# pip/setuptools; downgrade setuptools first, then install the rest:
pip install "setuptools<60"
pip install autocorrect==2.6.1
pip install -r requirements.txt

cp .env.example .env
# edit .env: set FLOWISE_API_URL (and FLOWISE_API_KEY if your chatflow requires it)
```

## Prepare your QA dataset

Edit `data/qa_dataset.example.csv` (or create your own CSV) with your own
questions and the *expected* answers from your RAG knowledge base:

```csv
question,answer
What is our refund policy?,Refunds are issued within 14 days of purchase.
```

promptbench's built-in datasets (SQuAD, GLUE, etc.) are generic public
benchmarks — they won't reflect a Flowise chatflow's actual knowledge base,
so this tool is designed around your own domain-specific Q&A pairs instead.

## Run

```bash
python eval_rag.py \
  --url https://cloud.flowiseai.com/api/v1/prediction/d8c9773e-d2f0-4045-89a3-667f2ec75559 \
  --dataset data/qa_dataset.example.csv \
  --output results.csv
```

(If `FLOWISE_API_URL`/`FLOWISE_API_KEY` are set in `.env`, `--url`/`--api-key`
can be omitted.)

This prints per-prompt-template EM/F1 plus an overall average, and writes:

- `results.csv` — one row per (template, question) with the raw/cleaned
  answer, EM, F1, latency, and any error.
- `results.summary.json` — aggregated stats per template + overall.

Useful flags: `--limit N` (quick smoke test on N rows), `--workers N`
(parallel requests), `--sleep S` (throttle sequential requests),
`--templates path.txt` (one custom prompt template per line, each
containing `{question}`).

## Tests

No network access needed — `requests` and the Flowise HTTP calls are
mocked:

```bash
pytest tests/
```
