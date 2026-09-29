"""Tests sur le vrai corpus (data/corpus.jsonl). Ignorés si le corpus n'est pas construit."""
import json
import math
from pathlib import Path

import pytest

from rag.bm25 import score_all, search
from rag.index import build_index
from rag.tokenizer import tokenize

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus.jsonl"
pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="lancer scripts/build_corpus.py d'abord")


@pytest.fixture(scope="module")
def docs():
    return [json.loads(l) for l in CORPUS.open(encoding="utf-8")]


@pytest.fixture(scope="module")
def index(docs):
    return build_index([d["text"] for d in docs])


def test_corpus_has_at_least_1000_documents(docs):
    assert len(docs) >= 1000


def test_corpus_has_three_sources(docs):
    assert {d["source"] for d in docs} == {"fastapi", "starlette", "pydantic"}


def test_index_invariants(docs, index):
    assert index.n_docs == len(docs)
    total_from_postings = sum(f for p in index.postings.values() for f in p.values())
    assert total_from_postings == sum(index.doc_len)          # aucun token perdu ni compté deux fois
    assert index.avgdl == pytest.approx(sum(index.doc_len) / len(docs))


def test_rebuild_is_deterministic(docs, index):
    again = build_index([d["text"] for d in docs])
    assert again.postings == index.postings
    assert again.doc_len == index.doc_len


def naive_bm25(texts, query, k1=1.2, b=0.75):
    """BM25 sans index, écrit indépendamment : on rescanne tout le corpus pour chaque requête."""
    toks = [tokenize(t) for t in texts]
    N = len(toks)
    avgdl = sum(len(t) for t in toks) / N
    scores = {}
    for term in set(tokenize(query)):
        n_t = sum(1 for t in toks if term in t)
        if n_t == 0:
            continue
        idf = math.log(1 + (N - n_t + 0.5) / (n_t + 0.5))
        for i, t in enumerate(toks):
            f = t.count(term)
            if f:
                scores[i] = scores.get(i, 0.0) + idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / avgdl))
    return scores


@pytest.mark.parametrize("query", [
    "dependency injection", "validate request body", "field validator", "websocket",
    "how do I run background tasks after returning a response", "CORS middleware",
    "model_dump json", "zzzunknownterm",
])
def test_indexed_bm25_matches_naive_full_scan(docs, index, query):
    texts = [d["text"] for d in docs]
    fast, slow = score_all(index, query), naive_bm25(texts, query)
    assert fast.keys() == slow.keys()
    for doc_id in slow:
        assert fast[doc_id] == pytest.approx(slow[doc_id])


@pytest.mark.parametrize("k1,b", [(1.2, 0.75), (2.0, 0.3), (0.5, 1.0), (1.5, 0.0)])
def test_naive_equivalence_across_parameters(docs, index, k1, b):
    texts = [d["text"] for d in docs]
    fast, slow = score_all(index, "custom exception handler", k1, b), naive_bm25(texts, "custom exception handler", k1, b)
    for doc_id in slow:
        assert fast[doc_id] == pytest.approx(slow[doc_id])


# Les 5 requêtes de contrôle du gate : pour chacune, la page attendue doit apparaître dans le top-5.
GATE_QUERIES = [
    ("dependency injection", {"fastapi/tutorial/dependencies/index.md", "fastapi/features.md"}),
    ("validate request body", {"fastapi/tutorial/body.md"}),
    ("custom exception handler", {"fastapi/tutorial/handling-errors.md"}),
    ("background tasks", {"fastapi/tutorial/background-tasks.md", "starlette/background.md"}),
    ("field validator", {"pydantic/concepts/validators.md"}),
]


@pytest.mark.parametrize("query,expected_pages", GATE_QUERIES)
def test_gate_query_returns_expected_page_in_top5(docs, index, query, expected_pages):
    top_pages = {docs[i]["path"] for i, _ in search(index, query, k=5)}
    assert top_pages & expected_pages, f"{query!r}: {top_pages}"


def test_scores_are_sorted_descending(docs, index):
    res = search(index, "dependency injection", k=20)
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)
