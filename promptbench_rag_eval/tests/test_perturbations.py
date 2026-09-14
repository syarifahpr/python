import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturbations import PERTURBATIONS, checklist, deepwordbug, keyboard


def test_perturbations_registry_has_expected_attacks():
    assert set(PERTURBATIONS) == {"deepwordbug", "keyboard", "checklist"}


def test_deepwordbug_is_deterministic_given_seed():
    text = "Apa syarat iklim yang cocok untuk budidaya sorgum?"
    out1 = deepwordbug(text, seed=42)
    out2 = deepwordbug(text, seed=42)
    assert out1 == out2


def test_deepwordbug_changes_text_with_high_rate():
    text = "Bagaimana cara mempersiapkan lahan sebelum menanam sorgum dengan baik"
    out = deepwordbug(text, rate=1.0, seed=1)
    assert out != text
    # word count should be preserved (only in-word edits, no word deletion)
    assert len(out.split(" ")) == len(text.split(" "))


def test_keyboard_is_deterministic_given_seed():
    text = "Berapa jarak tanam yang direkomendasikan untuk sorgum"
    out1 = keyboard(text, seed=7)
    out2 = keyboard(text, seed=7)
    assert out1 == out2


def test_keyboard_preserves_length():
    text = "sorgum tahan kekeringan"
    out = keyboard(text, rate=1.0, seed=3)
    assert len(out) == len(text)


def test_checklist_is_deterministic_given_seed():
    text = "Apa itu sorgum?"
    out1 = checklist(text, seed=5)
    out2 = checklist(text, seed=5)
    assert out1 == out2


def test_checklist_strips_trailing_question_mark_or_lowercases():
    text = "Apa itu sorgum?"
    out = checklist(text, seed=5)
    assert out != text
