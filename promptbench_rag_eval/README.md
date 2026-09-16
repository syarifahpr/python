# PromptBench RAG Evaluation

Evaluates a RAG chatbot's API (Flowise or Dify) by sending it a set of
labeled question/answer pairs under several prompt phrasings, and scoring
the answers with exact-match / F1 — the same idea behind
[promptbench](https://github.com/microsoft/promptbench)'s prompt
robustness evaluations, applied to a live RAG API instead of a raw LLM.
`--provider` selects the backend; both talk to the same `eval_rag.py` /
`eval_robustness.py` scripts and the same QA dataset format.

A third script, `eval_deepchecks.py`, adds a complementary check using
[deepchecks](https://github.com/deepchecks/deepchecks)' NLP module — see
"Text-quality checks with deepchecks" below.

The chatflow originally under test in this repo's defaults is the public
Flowise chatbot at
https://cloud.flowiseai.com/chatbot/d8c9773e-d2f0-4045-89a3-667f2ec75559
(a RAG assistant about sorghum/*sorgum* cultivation) — its underlying
prediction endpoint is
`https://cloud.flowiseai.com/api/v1/prediction/d8c9773e-d2f0-4045-89a3-667f2ec75559`
(same chatflow ID, `/api/v1/prediction/` instead of `/chatbot/`), which is
what `--provider flowise` calls. `data/qa_dataset.example.csv` contains
Q&A pairs about sorghum cultivation to match.

A second chat app is also supported, on [Dify](https://dify.ai), via
`--provider dify`: https://udify.app/chat/4KeH0H4I0KUVfUJ3 — **that
share link is Dify's human chat UI, not a callable API.** To evaluate it
programmatically you need an API key from the Dify console (open the app
-> "API Access"/"Akses API" -> API Key); the app is identified by that key,
not by the udify.app URL. See `dify_client.py` for details.

## How it fits together

- `flowise_client.py` — thin REST client for Flowise's `POST {url}` with
  `{"question": ...}`, with retries and defensive response parsing.
- `dify_client.py` — thin REST client for Dify's `POST {base_url}/chat-messages`
  with `{"query": ..., "response_mode": "blocking", ...}` and a Bearer API
  key, same retry behavior as the Flowise client.
- `providers.py` — picks Flowise or Dify based on `--provider` and builds
  the right client.
- `rag_model.py` — wraps either client as a `model(input_text) -> str`
  callable, the same calling convention promptbench's own `LLMModel` uses,
  so promptbench's prompt/templating utilities can be reused unmodified.
- `errors.py` — shared quota/rate-limit error detection used by both
  clients and both eval scripts.
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
- `dataset.py` — shared CSV loader used by `eval_rag.py`, `eval_robustness.py`,
  and `eval_deepchecks.py`.
- `eval_deepchecks.py` — a third evaluation mode using deepchecks' NLP
  checks instead of EM/F1, for text-quality/integrity issues EM/F1 doesn't
  catch (duplicate/canned answers, garbled output, property drift). See
  "Text-quality checks with deepchecks" below. Deliberately does not import
  promptbench (see that section for why).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate

# promptbench's dependency `autocorrect==2.6.1` fails to build on modern
# pip/setuptools; downgrade setuptools first, then install the rest:
pip install "setuptools<60"
pip install autocorrect==2.6.1
pip install -r requirements.txt

cp .env.example .env
# edit .env: set FLOWISE_API_URL (and FLOWISE_API_KEY if your chatflow requires it),
# or DIFY_API_KEY if you're using --provider dify
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

Flowise:
```bash
python eval_rag.py --provider flowise \
  --url https://cloud.flowiseai.com/api/v1/prediction/d8c9773e-d2f0-4045-89a3-667f2ec75559 \
  --dataset data/qa_dataset.example.csv \
  --output results.csv
```

Dify (get the API key from the Dify console first — see the intro above):
```bash
python eval_rag.py --provider dify --api-key app-xxxxxxxx \
  --dataset data/qa_dataset.example.csv \
  --output results.csv
```

`--provider` defaults to `flowise`. If `FLOWISE_API_URL`/`FLOWISE_API_KEY`
or `DIFY_API_KEY`/`DIFY_BASE_URL` are set in `.env`, `--url`/`--api-key`
can be omitted.

This prints per-prompt-template EM/F1 plus an overall average, and writes:

- `results.csv` — one row per (template, question) with the raw/cleaned
  answer, EM, F1, latency, and any error.
- `results.summary.json` — aggregated stats per template + overall.

Useful flags: `--limit N` (quick smoke test on N rows), `--workers N`
(parallel requests), `--sleep S` (throttle sequential requests),
`--templates path.txt` (one custom prompt template per line, each
containing `{question}`).

If the provider's quota/rate limit is exhausted mid-run (Flowise's
`"Predictions limit exceeded"`, Dify's HTTP 429, or anything else
`errors.is_quota_error` recognizes), both `eval_rag.py` and
`eval_robustness.py` stop immediately (printing what happened and saving
whatever results were already collected) instead of sending the remaining
requests, which would all fail identically.

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
python eval_robustness.py --provider flowise \
  --url https://cloud.flowiseai.com/api/v1/prediction/d8c9773e-d2f0-4045-89a3-667f2ec75559 \
  --dataset data/qa_dataset.example.csv \
  --output robustness.csv \
  --limit 5
```
(or `--provider dify --api-key app-xxxxxxxx`, same as above)

For each question this sends `1 + len(attacks)` requests (clean +
`deepwordbug` + `keyboard` + `checklist` by default — pick a subset with
`--attacks deepwordbug,keyboard`), so quota is consumed faster than
`eval_rag.py`; start with `--limit` on a small QA set. Defaults to
`--sleep 1.0` between requests since free-tier quotas tend to be low
(see "Run" above).

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

## Text-quality checks with deepchecks

EM/F1 (above) only tells you whether an answer matches the expected
ground truth. It won't catch a chatbot that returns the same canned
fallback ("I don't know") for many different questions, garbled/encoding-broken
output, or answers whose overall character start drifting once the input
is slightly perturbed. [deepchecks](https://github.com/deepchecks/deepchecks)'
NLP module has checks built specifically for that, so `eval_deepchecks.py`
collects the chatbot's answers the same way `eval_rag.py` does and runs them
through:

- `TextDuplicates` — flags when too many answers are near-duplicates of
  each other (a common RAG failure mode: the same fallback/boilerplate
  response for unrelated questions).
- `SpecialCharacters` — flags answers with an abnormal amount of special
  characters (garbled/encoding-broken output).
- `FrequentSubstrings` — flags repeated boilerplate substrings across
  answers.
- `TextPropertyOutliers` — flags answers that are statistical outliers on
  properties like length, so unusually short/long/off answers stand out.
- With `--compare-attack`, also runs `TrainTestSamplesMix` and
  `PropertyDrift` comparing the clean-question answers against answers to
  a perturbed version of each question (see "Robustness testing" above),
  to catch cases where perturbing the question causes a qualitatively
  different kind of answer (not just a lower EM/F1 score).

This is a different dependency stack from `eval_rag.py`/`eval_robustness.py`
and deliberately does not import promptbench: promptbench's own answer
cleanup (`OutputProcess.general`) lowercases and strips punctuation, which
is fine for EM/F1 matching but would corrupt the text deepchecks is meant
to inspect. Install it separately:

```bash
pip install -r requirements-deepchecks.txt
```

`requirements-deepchecks.txt` pins `scikit-learn`/`category-encoders` to
versions that actually work with `deepchecks==0.19.1` — that package
doesn't cap either itself, so a plain `pip install deepchecks[nlp]` can
resolve versions of both that break deepchecks' own imports.

```bash
python eval_deepchecks.py --provider flowise \
  --url https://cloud.flowiseai.com/api/v1/prediction/d8c9773e-d2f0-4045-89a3-667f2ec75559 \
  --dataset data/qa_dataset.example.csv \
  --output deepchecks_report.html
```

(or `--provider dify --api-key app-xxxxxxxx`, same as above). Add
`--compare-attack deepwordbug` (or `keyboard`/`checklist`) to also collect
perturbed-question answers and run the drift comparison.

This prints each check's condition result (PASS/WARN/FAIL) and writes:

- `deepchecks_report.html` — the full interactive deepchecks report (open
  it in a browser).
- `deepchecks_report.csv` — the raw answers collected, with EM/F1 for
  reference.
- `deepchecks_report.summary.json` — the same PASS/WARN/FAIL conditions
  printed to the console, as JSON.
- With `--compare-attack`, also `deepchecks_report.compare-<attack>.html`
  and `.compare-<attack>.csv` for the perturbed-answer comparison.

**First-run model download:** the property-based checks (`TextPropertyOutliers`,
`PropertyDrift`) call deepchecks' `TextData.calculate_builtin_properties()`,
which unconditionally downloads a ~130MB fastText language-ID model from
`dl.fbaipublicfiles.com` the first time it runs (cached afterwards) and
requires the `fasttext` package (included in `requirements-deepchecks.txt`,
but it needs a C++ compiler to build). If you're offline, don't have
`fasttext`, or don't want the download, pass `--skip-properties` — you
still get the duplicate/special-character/frequent-substrings/samples-mix
checks, just not the property-based ones.

## Tests

No network access needed — `requests` and the Flowise/Dify HTTP calls are
mocked:

```bash
pytest tests/
```
