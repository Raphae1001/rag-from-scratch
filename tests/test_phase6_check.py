import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from phase6_check import dense_search_in_memory  # noqa: E402


def test_dense_search_in_memory_picks_best_chunk_per_document():
    # doc 0 a 2 passages (le second plus proche de la requête), doc 1 en a 1
    chunk_doc_ids = [0, 0, 1]
    chunk_vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    query = np.array([0.0, 1.0])  # aligné avec le 2e passage du doc 0
    result = dense_search_in_memory(query, chunk_doc_ids, chunk_vectors, k=2)
    assert result[0] == 0  # doc 0 en premier grâce à son meilleur passage (score 1.0)
    assert set(result) == {0, 1}


def test_dense_search_in_memory_respects_k():
    chunk_doc_ids = [0, 1, 2, 3]
    chunk_vectors = np.eye(4)
    query = np.array([1.0, 0.0, 0.0, 0.0])
    result = dense_search_in_memory(query, chunk_doc_ids, chunk_vectors, k=2)
    assert len(result) == 2
    assert result[0] == 0
