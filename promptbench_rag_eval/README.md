# PromptBench RAG Evaluation for Flowise

Evaluates a Flowise "prediction" API endpoint (a deployed RAG chatflow) by
sending it a set of labeled question/answer pairs under several prompt
phrasings, and scoring the answers with exact-match / F1 — the same idea
behind [promptbench](https://github.com/microsoft/promptbench)'s prompt
robustness evaluations, applied to a live RAG API instead of a raw LLM.

The chatflow under test in this repo's defaults is the public chatbot at
https://cloud.flowiseai.com/chatbot/d8c9773e-d2f0-4045-89a3-667f2ec75559
(a RAG assistant about sorghum/*sorgum* cultivation) — its underlying
prediction endpoint is
`https://cloud.flowiseai.com/api/v1/prediction/d8c9773e-d2f0-4045-89a3-667f2ec75559`
(same chatflow ID, `/api/v1/prediction/` instead of `/chatbot/`), which is
what `eval_rag.py` actually calls. `data/qa_dataset.example.csv` contains
Q&A pairs about sorghum cultivation to match.

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
- `perturbations.py` / `eval_robustness.py` — a second evaluation mode that
  measures robustness to small, meaning-preserving noise in the question
  (typos, casing/punctuation changes), rather than to different prompt
  phrasings. See "Robustness testing" below.
- `dataset.py` — shared CSV loader used by both `eval_rag.py` and
  `eval_robustness.py`.

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
The bundled `data/qa_dataset.example.csv` already has 20 sorghum-cultivation
Q&A pairs for the chatbot linked above; if that chatflow's source documents
say something different, update the `answer` column to match them so EM/F1
actually measures agreement with the chatbot's real knowledge base.

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

If the Flowise account's prediction quota is exhausted mid-run, both
`eval_rag.py` and `eval_robustness.py` detect the
`"Predictions limit exceeded"` error and stop immediately (printing what
happened and saving whatever results were already collected) instead of
sending the remaining requests, which would all fail identically.

## Robustness testing

promptbench's own adversarial attacks (TextFooler, TextBugger, DeepWordBug,
BERTAttack, CheckList, StressTest, in `promptbench.prompt_attack`) are
built on the `textattack` library and target classification tasks with a
fixed label set — their "goal function" needs a discrete label to flip,
which doesn't apply to a RAG chatbot's free-text answers. `eval_robustness.py`
implements the same underlying idea directly: for each question it sends
the clean version plus a few perturbed versions (small typos, keyboard
mistakes, casing/punctuation changes — see `perturbations.py`) and measures
how much the answer degrades.

```bash
python eval_robustness.py \
  --url https://cloud.flowiseai.com/api/v1/prediction/d8c9773e-d2f0-4045-89a3-667f2ec75559 \
  --dataset data/qa_dataset.example.csv \
  --output robustness.csv \
  --limit 5
```

For each question this sends `1 + len(attacks)` requests (clean +
`deepwordbug` + `keyboard` + `checklist` by default — pick a subset with
`--attacks deepwordbug,keyboard`), so quota is consumed faster than
`eval_rag.py`; start with `--limit` on a small QA set. Defaults to
`--sleep 1.0` between requests since Flowise Cloud's free tier has a low
prediction quota (see "Run" above).

It prints, per attack type:

- `avg_f1` — average EM/F1-style word-overlap score against the ground
  truth answer for that attack's perturbed questions.
- `f1_drop_vs_clean` — how much lower that is than the clean-question
  baseline. Higher means the chatbot is more sensitive to that kind of
  noise (less robust).
- `avg_answer_stability_vs_clean` — word overlap between the perturbed
  answer and the clean answer for the *same* question, regardless of
  correctness. Lower means the chatbot's response itself changed more,
  even if both answers happened to be equally right or wrong.

`robustness.csv` has one row per (question, attack) with the perturbed
question text and the answer actually returned, and
`robustness.summary.json` has the aggregated stats above.

## Tests

No network access needed — `requests` and the Flowise HTTP calls are
mocked:

```bash
pytest tests/
```
