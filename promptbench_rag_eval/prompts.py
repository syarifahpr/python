"""Default prompt templates used to probe the RAG chatflow's robustness to
phrasing, in the spirit of promptbench's prompt-engineering evaluations.

Each template must contain a ``{question}`` placeholder — it is filled via
``promptbench.utils.InputProcess.basic_format``.
"""

DEFAULT_TEMPLATES = [
    "{question}",
    "Please answer the following question as accurately as possible: {question}",
    "Answer this question concisely and only using information you are confident about: {question}",
    "You are a helpful assistant answering questions about our knowledge base.\nQuestion: {question}\nAnswer:",
]


def load_templates(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_TEMPLATES
    with open(path, "r", encoding="utf-8") as f:
        templates = [line.strip() for line in f if line.strip()]
    if not templates:
        raise ValueError(f"No templates found in {path}")
    for t in templates:
        if "{question}" not in t:
            raise ValueError(f"Template missing '{{question}}' placeholder: {t!r}")
    return templates
