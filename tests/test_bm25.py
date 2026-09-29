import math

import pytest

from rag.bm25 import idf, score_all, search, term_score
from rag.index import build_index

# Mini-corpus jouet (voir test_index.py) : N = 3, avgdl = 10/3, k1 = 1.2, b = 0.75
TEXTS = [
    "apple banana apple",
    "banana cherry",
    "apple cherry cherry cherry date",
]


def test_idf_by_hand():
    idx = build_index(TEXTS)
    # apple : n_t = 2 -> ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6)
    assert idf(idx, "apple") == pytest.approx(math.log(1.6))
    # date : n_t = 1 -> ln(1 + (3 - 1 + 0.5) / (1 + 0.5)) = ln(1 + 2.5/1.5)
    assert idf(idx, "date") == pytest.approx(math.log(1 + 2.5 / 1.5))
    # un terme rare pèse plus qu'un terme fréquent
    assert idf(idx, "date") > idf(idx, "apple")


def test_term_score_by_hand():
    # d0, terme "apple" : f = 2, |D| = 3
    #   norm = 1 - 0.75 + 0.75 * 3 / (10/3) = 0.925
    #   tf   = 2 * 2.2 / (2 + 1.2 * 0.925) = 4.4 / 3.11
    assert term_score(2, 3, 10 / 3, 1.2, 0.75) == pytest.approx(4.4 / 3.11)
    # d2, terme "apple" : f = 1, |D| = 5 -> norm = 1.375 -> tf = 2.2 / (1 + 1.65)
    assert term_score(1, 5, 10 / 3, 1.2, 0.75) == pytest.approx(2.2 / 2.65)


def test_bm25_single_term_regression():
    """Non-régression : scores calculés à la main pour la requête "apple"."""
    idx = build_index(TEXTS)
    scores = score_all(idx, "apple")
    assert set(scores) == {0, 2}                       # d1 ne contient pas "apple"
    assert scores[0] == pytest.approx(math.log(1.6) * 4.4 / 3.11)   # ~ 0.66496
    assert scores[2] == pytest.approx(math.log(1.6) * 2.2 / 2.65)   # ~ 0.39019


def test_bm25_multi_term_regression():
    """Requête "apple cherry" : le score est la somme des scores par terme."""
    idx = build_index(TEXTS)
    w = math.log(1.6)  # apple et cherry ont tous deux n_t = 2
    scores = score_all(idx, "apple cherry")
    assert scores[0] == pytest.approx(w * 4.4 / 3.11)                       # apple seul
    assert scores[1] == pytest.approx(w * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 2 / (10 / 3))))  # cherry, |D|=2
    assert scores[2] == pytest.approx(w * 2.2 / 2.65 + w * 3 * 2.2 / (3 + 1.2 * 1.375))  # apple + cherry (f=3)


def test_search_ranking_and_topk():
    idx = build_index(TEXTS)
    res = search(idx, "apple cherry", k=2)
    assert [d for d, _ in res] == [2, 0]               # d2 (les deux termes) puis d0
    assert len(res) == 2
    assert res[0][1] > res[1][1]


def test_saturation():
    # 10 occurrences ne valent pas 10x une occurrence
    one = term_score(1, 100, 100, 1.2, 0.75)
    ten = term_score(10, 100, 100, 1.2, 0.75)
    assert one < ten < 10 * one
    assert ten < 1.2 + 1                                # plafond k1 + 1


def test_length_normalization():
    # même f, document plus long -> score plus faible (b > 0)
    assert term_score(2, 200, 100, 1.2, 0.75) < term_score(2, 50, 100, 1.2, 0.75)
    # b = 0 : la longueur n'a plus d'effet
    assert term_score(2, 200, 100, 1.2, 0.0) == term_score(2, 50, 100, 1.2, 0.0)


def test_k1_b_are_configurable():
    idx = build_index(TEXTS)
    a = score_all(idx, "apple", k1=1.2, b=0.75)[0]
    c = score_all(idx, "apple", k1=2.0, b=0.3)[0]
    assert a != c


def test_edge_cases():
    idx = build_index(TEXTS)
    assert search(idx, "") == []
    assert search(idx, "zzzunknown") == []
    assert len(search(idx, "apple", k=50)) == 2         # k > nb de résultats
    assert score_all(idx, "apple apple") == score_all(idx, "apple")  # terme répété = compté une fois
