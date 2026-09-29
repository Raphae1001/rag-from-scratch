import numpy as np
import pytest

from rag.chunking import chunk_words, make_chunks
from rag.embed import HashingEmbedder
from rag.fusion import reciprocal_rank_fusion
from rag.metrics import precision_at_k, recall_at_k, reciprocal_rank


# --- métriques -------------------------------------------------------------------------------
def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c", "z"}, 3) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "b", "c"], {"a", "c", "z"}, 1) == pytest.approx(1 / 3)
    assert recall_at_k([], {"a"}, 10) == 0.0


def test_recall_requires_annotation():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 5)


def test_precision_at_k_uses_k_as_denominator():
    assert precision_at_k(["a", "x"], {"a"}, 5) == pytest.approx(1 / 5)


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x"], {"a"}) == 0.0


# --- fusion RRF ------------------------------------------------------------------------------
def test_rrf_by_hand():
    # classement 1 : 10, 20, 30 ; classement 2 : 20, 40
    res = dict(reciprocal_rank_fusion([[10, 20, 30], [20, 40]], k=60))
    assert res[10] == pytest.approx(1 / 61)
    assert res[20] == pytest.approx(1 / 62 + 1 / 61)
    assert res[30] == pytest.approx(1 / 63)
    assert res[40] == pytest.approx(1 / 62)


def test_rrf_order_and_document_in_both_lists_wins():
    order = [d for d, _ in reciprocal_rank_fusion([[1, 2, 3], [3, 4, 5]])]
    assert order[0] == 3          # présent dans les deux listes (rang 3 et rang 1)
    assert set(order) == {1, 2, 3, 4, 5}


def test_rrf_is_deterministic_on_ties():
    assert [d for d, _ in reciprocal_rank_fusion([[7], [3]])] == [3, 7]


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# --- découpage -------------------------------------------------------------------------------
def test_short_text_is_one_chunk():
    assert chunk_words("a b c", size=10, overlap=2) == ["a b c"]
    assert chunk_words("", size=10, overlap=2) == []


def test_chunks_overlap_and_cover_everything():
    text = " ".join(str(i) for i in range(50))
    chunks = chunk_words(text, size=20, overlap=5)
    assert chunks[0].split()[:2] == ["0", "1"]
    assert all(len(c.split()) <= 20 for c in chunks)
    # chevauchement : les 5 derniers mots d'un chunk ouvrent le suivant
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-5:] == b.split()[:5]
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(text.split())              # aucun mot perdu
    assert chunks[-1].split()[-1] == "49"


def test_chunk_size_must_exceed_overlap():
    with pytest.raises(ValueError):
        chunk_words("a b", size=5, overlap=5)


def test_make_chunks_adds_context_to_later_chunks_only():
    doc = {"page": "Handling Errors", "section": "Custom handlers", "text": " ".join(["w"] * 300)}
    chunks = make_chunks(doc, size=100, overlap=10)
    assert len(chunks) > 1
    assert not chunks[0].startswith("Handling Errors")
    assert all(c.startswith("Handling Errors - Custom handlers: ") for c in chunks[1:])


# --- faux embedder ---------------------------------------------------------------------------
def test_hashing_embedder_is_deterministic_and_normalized():
    e = HashingEmbedder()
    a, b = e.encode(["background tasks", "background tasks"])
    assert np.allclose(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert e.encode(["!!!"])[0].tolist() == [0.0] * e.dim      # aucun token -> vecteur nul, pas de NaN


def test_hashing_embedder_similarity_reflects_word_overlap():
    e = HashingEmbedder()
    q, close, far = e.encode(["custom exception handler", "install a custom exception handler", "gzip compression"])
    assert float(q @ close) > float(q @ far)
