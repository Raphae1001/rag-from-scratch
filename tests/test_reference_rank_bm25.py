"""Contrôle indépendant : comparaison à la bibliothèque de référence rank_bm25.

rank_bm25 utilise la même partie TF/longueur/avgdl que nous, mais une IDF différente (variante ATIRE
avec plancher). On vérifie donc deux choses :
1. avec l'IDF alignée sur la nôtre (Lucene), les scores sont identiques au flottant près ;
2. avec son IDF native, le classement reste quasi identique (même top-1, ≥ 8/10 résultats communs).
"""
import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
rank_bm25 = pytest.importorskip("rank_bm25")

from rag.bm25 import score_all, search  # noqa: E402
from rag.index import build_index  # noqa: E402
from rag.tokenizer import tokenize  # noqa: E402

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus.jsonl"
pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="lancer scripts/build_corpus.py d'abord")

QUERIES = [
    "dependency injection", "validate request body", "custom exception handler", "background tasks",
    "field validator", "websocket", "CORS middleware", "model_dump json",
    "how do I run background tasks after returning a response", "response model", "path parameters",
]


class LuceneOkapi(rank_bm25.BM25Okapi):
    def _calc_idf(self, nd):
        for term, n_t in nd.items():
            self.idf[term] = math.log(1 + (self.corpus_size - n_t + 0.5) / (n_t + 0.5))


@pytest.fixture(scope="module")
def setup():
    docs = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
    texts = [d["text"] for d in docs]
    toks = [tokenize(t) for t in texts]
    return len(docs), toks, build_index(texts)


@pytest.mark.parametrize("query", QUERIES)
def test_scores_equal_reference_with_aligned_idf(setup, query):
    n, toks, idx = setup
    ref = LuceneOkapi(toks, k1=1.2, b=0.75).get_scores(list(dict.fromkeys(tokenize(query))))
    mine = score_all(idx, query)
    for i in range(n):
        assert mine.get(i, 0.0) == pytest.approx(ref[i], abs=1e-9)


@pytest.mark.parametrize("query", QUERIES)
def test_ranking_close_to_reference_with_native_idf(setup, query):
    n, toks, idx = setup
    native = rank_bm25.BM25Okapi(toks, k1=1.2, b=0.75).get_scores(list(dict.fromkeys(tokenize(query))))
    ref_top = [int(i) for i in np.argsort(-native, kind="stable")[:10]]
    my_top = [d for d, _ in search(idx, query, k=10)]
    assert my_top[0] == ref_top[0]
    assert len(set(my_top) & set(ref_top)) >= 8
