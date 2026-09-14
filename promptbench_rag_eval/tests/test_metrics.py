import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import aggregate, exact_match_score, f1_score, normalize_answer, score_pair


def test_normalize_answer_strips_punctuation_articles_and_case():
    assert normalize_answer("The Flowise API!") == "flowise api"
    assert normalize_answer("  extra   spaces  ") == "extra spaces"


def test_exact_match_score():
    assert exact_match_score("Paris", "paris") == 1.0
    assert exact_match_score("The Paris", "Paris") == 1.0
    assert exact_match_score("Lyon", "Paris") == 0.0


def test_f1_score_partial_overlap():
    score = f1_score("Paris is the capital", "Paris is the capital of France")
    assert 0.0 < score < 1.0


def test_f1_score_no_overlap_is_zero():
    assert f1_score("banana", "Paris") == 0.0


def test_f1_score_empty_prediction():
    assert f1_score("", "Paris") == 0.0
    assert f1_score("", "") == 1.0


def test_score_pair_returns_em_and_f1():
    result = score_pair("Paris", "Paris")
    assert result == {"em": 1.0, "f1": 1.0}


def test_aggregate_averages_scores():
    scores = [{"em": 1.0, "f1": 1.0}, {"em": 0.0, "f1": 0.5}]
    result = aggregate(scores)
    assert result["em"] == 0.5
    assert result["f1"] == 0.75
    assert result["n"] == 2


def test_aggregate_empty_list():
    assert aggregate([]) == {"em": 0.0, "f1": 0.0, "n": 0}
