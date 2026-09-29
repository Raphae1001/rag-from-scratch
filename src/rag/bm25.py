"""IDF + scoring BM25 + recherche top-k. (J2-J3)

score(D, q) = sum_t IDF(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * |D| / avgdl))

IDF(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))   (variante Lucene : toujours >= 0)
"""
import heapq
import math

from rag.index import InvertedIndex
from rag.tokenizer import tokenize


def idf(index: InvertedIndex, term: str) -> float:
    n_t = index.doc_freq(term)
    return math.log(1 + (index.n_docs - n_t + 0.5) / (n_t + 0.5))


def term_score(f: int, doc_len: int, avgdl: float, k1: float, b: float) -> float:
    """Partie TF saturée et normalisée par la longueur (sans l'IDF)."""
    norm = 1 - b + b * doc_len / avgdl
    return f * (k1 + 1) / (f + k1 * norm)


def score_all(index: InvertedIndex, query: str, k1: float = 1.2, b: float = 0.75) -> dict[int, float]:
    """Score de tous les documents contenant au moins un terme de la requête."""
    scores: dict[int, float] = {}
    # set() : un terme répété dans la requête ne compte qu'une fois
    for term in set(tokenize(query)):
        postings = index.postings.get(term)
        if not postings:
            continue
        w = idf(index, term)
        for doc_id, f in postings.items():
            s = w * term_score(f, index.doc_len[doc_id], index.avgdl, k1, b)
            scores[doc_id] = scores.get(doc_id, 0.0) + s
    return scores


def search(index: InvertedIndex, query: str, k: int = 10, k1: float = 1.2, b: float = 0.75) -> list[tuple[int, float]]:
    """Top-k classé : [(doc_id, score), ...]. Égalités départagées par doc_id croissant."""
    scores = score_all(index, query, k1, b)
    return heapq.nsmallest(k, scores.items(), key=lambda kv: (-kv[1], kv[0]))
