"""Lightweight, meaning-preserving text perturbations for robustness testing.

promptbench's own adversarial attacks (TextFooler, TextBugger, DeepWordBug,
BERTAttack, CheckList, StressTest, in `promptbench.prompt_attack`) are built
on the `textattack` library and target classification tasks with a fixed
label set (their goal function needs a discrete label to flip). A RAG
chatbot's answer is free text, not a label, so that machinery doesn't apply
directly here.

This module implements the same underlying idea — perturb the input in ways
that shouldn't change its meaning to a human reader, then see how much the
model's answer degrades — with simple, dependency-free, seeded functions
named after the promptbench/TextAttack recipes they're inspired by:

- ``deepwordbug``: character-level noise (swap/insert/delete/substitute a
  few characters within words), like DeepWordBug's edit-distance attacks.
- ``checklist``: casing and punctuation perturbations (as used in the
  CheckList behavioral testing suite), e.g. dropping the question mark or
  randomly capitalizing words.
- ``keyboard``: characters replaced by their neighbor on a QWERTY keyboard,
  simulating realistic fat-finger typos.
"""
from __future__ import annotations

import random
import string

QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
    "a": "qsz", "s": "awd", "d": "sef", "f": "dgr", "g": "fht", "h": "gjy",
    "j": "hku", "k": "jli", "l": "ko",
    "z": "as", "x": "zc", "c": "xv", "v": "cb", "b": "vn", "n": "bm", "m": "n",
}


def deepwordbug(text: str, rate: float = 0.15, seed: int | None = None) -> str:
    """Randomly swap, delete, insert, or substitute a character within a
    handful of words (character-level noise), mimicking DeepWordBug."""
    rng = random.Random(seed)
    words = text.split(" ")
    for idx, word in enumerate(words):
        if len(word) < 4 or rng.random() > rate:
            continue
        pos = rng.randrange(1, len(word) - 1)
        op = rng.choice(["swap", "delete", "insert", "substitute"])
        if op == "swap":
            chars = list(word)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            word = "".join(chars)
        elif op == "delete":
            word = word[:pos] + word[pos + 1:]
        elif op == "insert":
            word = word[:pos] + rng.choice(string.ascii_lowercase) + word[pos:]
        elif op == "substitute":
            word = word[:pos] + rng.choice(string.ascii_lowercase) + word[pos + 1:]
        words[idx] = word
    return " ".join(words)


def keyboard(text: str, rate: float = 0.1, seed: int | None = None) -> str:
    """Replace a few characters with an adjacent QWERTY key, simulating
    realistic typing mistakes."""
    rng = random.Random(seed)
    chars = list(text)
    for i, ch in enumerate(chars):
        lower = ch.lower()
        if lower in QWERTY_NEIGHBORS and rng.random() < rate:
            replacement = rng.choice(QWERTY_NEIGHBORS[lower])
            chars[i] = replacement.upper() if ch.isupper() else replacement
    return "".join(chars)


def checklist(text: str, seed: int | None = None) -> str:
    """Casing/punctuation perturbations in the spirit of the CheckList
    behavioral testing suite: drop trailing punctuation, randomly
    capitalize a word, and collapse/duplicate a little whitespace."""
    rng = random.Random(seed)
    result = text.rstrip("?.! ")

    words = result.split(" ")
    if words:
        i = rng.randrange(len(words))
        words[i] = words[i].upper() if rng.random() < 0.5 else words[i].lower()
    result = " ".join(words)

    if rng.random() < 0.5:
        result = result.lower()

    if rng.random() < 0.3:
        pos = rng.randrange(len(result)) if result else 0
        result = result[:pos] + "  " + result[pos:]

    return result


PERTURBATIONS = {
    "deepwordbug": deepwordbug,
    "keyboard": keyboard,
    "checklist": checklist,
}
