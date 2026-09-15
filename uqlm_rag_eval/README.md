# uqlm RAG Hallucination Evaluation

Evaluates the Flowise chatbot's hallucination risk using
[uqlm](https://github.com/cvs-health/uqlm) (Uncertainty Quantification for
Language Models), against the same public chatflow this repo's sibling
`promptbench_rag_eval` project targets:
https://cloud.flowiseai.com/chatbot/d8c9773e-d2f0-4045-89a3-667f2ec75559
(a RAG assistant about sorghum/*sorgum* cultivation), via its prediction
endpoint
`https://cloud.flowiseai.com/api/v1/prediction/d8c9773e-d2f0-4045-89a3-667f2ec75559`.

Where `promptbench_rag_eval` scores *correctness* (EM/F1 against a known
answer), this tool scores *self-consistency*: for each question it asks the
chatbot the same thing several times and measures how much the answers
agree with each other. Low agreement is uqlm's signal that a response is
more likely to be a hallucination, independent of whether you have a
ground-truth answer at all. `data/qa_dataset.example.csv` (the same
sorghum Q&A set) is still used here so each question's uqlm confidence
score can also be cross-checked against an EM/F1 correctness score.

## How it fits together

- `flowise_client.py` — thin REST client for Flowise's
  `POST {base_url}/api/v1/prediction/{id}` with `{"question": ...}`,
  with retries and defensive response parsing (same client
  `promptbench_rag_eval` uses).
- `flowise_llm.py` — **the key adapter.** uqlm's scorers require their
  `llm` argument to be an actual `langchain_core.BaseChatModel` instance
  (`uqlm/utils/response_generator.py` asserts this and calls
  `await llm.ainvoke(messages)`), but Flowise's REST API isn't a LangChain
  integration. `FlowiseChatModel` subclasses `BaseChatModel` and forwards
  each call to `FlowiseClient.predict()`, using the last human message as
  the question (Flowise's endpoint takes a single `question` string, not a
  message history).
- `errors.py` / `dataset.py` / `metrics.py` — quota-error detection, CSV
  dataset loading, and SQuAD-style EM/F1 scoring, same as
  `promptbench_rag_eval`.
- `eval_uqlm.py` — CLI that builds a `FlowiseChatModel`, runs uqlm's
  `BlackBoxUQ.generate_and_score_sync(...)`, and writes per-question
  results + a summary.

## Why `BlackBoxUQ`, not `WhiteBoxUQ`

uqlm also ships `WhiteBoxUQ` (token log-probability-based scoring) and
`LLMPanel` (LLM-as-judge). `WhiteBoxUQ` needs the underlying model's token
log-probabilities, which a Flowise chatflow's REST response never exposes
(it only returns final text) — so `FlowiseChatModel` has no logprobs to
give it. `BlackBoxUQ`'s consistency-based scorers only need response text,
which is exactly what Flowise returns, so that's what this uses.
`LLMPanel` is not wired up here since it needs a second, judge LLM and its
own API key; it would be a reasonable extension if you have one available.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set FLOWISE_API_URL (and FLOWISE_API_KEY if your chatflow requires it)
```

**Note:** some scorers download models on first use — `semantic_negentropy`,
`noncontradiction`, `entailment`, and `semantic_sets_confidence` download an
NLI model (`microsoft/deberta-large-mnli`, ~1.5GB); `cosine_sim` downloads a
small sentence-transformer (~90MB); `bert_score` downloads a BERT model.
Make sure outbound network access to Hugging Face is available, or start
with `--scorers exact_match` (no downloads) to sanity-check connectivity to
Flowise first.

## Run

```bash
python eval_uqlm.py \
  --url https://cloud.flowiseai.com/api/v1/prediction/d8c9773e-d2f0-4045-89a3-667f2ec75559 \
  --dataset data/qa_dataset.example.csv \
  --output results.csv
```

If `FLOWISE_API_URL`/`FLOWISE_API_KEY` are set in `.env`, `--url`/`--api-key`
can be omitted.

Each question triggers `1 + num_responses` requests to Flowise (1 original
answer + `num_responses` sampled candidates used to measure consistency,
default 5), so a dataset of 20 questions makes 120 requests by default —
start with `--limit` on a small QA set if you're on a limited free-tier
quota.

Useful flags:

- `--num-responses N` — sampled responses per question for consistency
  scoring (uqlm's `num_responses`, default 5). Higher is more reliable but
  costs more requests.
- `--scorers a,b,c` — which uqlm `BlackBoxUQ` scorers to run. Default:
  `semantic_negentropy,noncontradiction,exact_match,cosine_sim` (uqlm's own
  default set). Full list: `semantic_negentropy`, `noncontradiction`,
  `exact_match`, `cosine_sim`, `bert_score`, `entailment`,
  `semantic_sets_confidence`.
- `--limit N` — only evaluate the first N rows of the dataset.
- `--timeout S` — per-request timeout in seconds.

This prints the average EM/F1 (correctness vs. `data/qa_dataset.example.csv`'s
`answer` column) plus the average of each uqlm confidence score (0 =
hallucination-prone/inconsistent, 1 = fully consistent across samples), and
writes:

- `results.csv` — one row per question with the response, EM/F1, each uqlm
  score, and the raw sampled candidate responses.
- `results.summary.json` — the aggregated averages plus uqlm's run
  metadata (temperature, sampling temperature, num_responses, scorers).

A question that scores low on `em`/`f1` *and* low on uqlm's confidence
scores is a real, self-consistent hallucination worth reviewing the
chatflow's source documents for; a question that scores low on `em`/`f1`
but *high* on uqlm's confidence scores is one where the chatbot is
consistently confident but consistently wrong (e.g. the ground-truth
answer in the CSV doesn't match what its knowledge base actually says).

## Tests

No network access needed — `requests` calls to Flowise are mocked, and
`test_eval_uqlm.py` runs uqlm's real `BlackBoxUQ` end-to-end against that
mock using only the `exact_match` scorer (no model downloads):

```bash
pytest tests/
```
