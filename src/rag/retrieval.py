"""Les trois configurations comparées dans les benchmarks : BM25, dense, hybride (RRF)."""
from rag.bm25 import search as bm25_search
from rag.db import dense_search
from rag.fusion import reciprocal_rank_fusion


def search_bm25(index, query: str, k: int = 10) -> list[int]:
    return [d for d, _ in bm25_search(index, query, k=k)]


def search_dense(conn, embedder, query: str, k: int = 10) -> list[int]:
    return [d for d, _ in dense_search(conn, embedder.encode([query])[0], k=k)]


def search_hybrid(index, conn, embedder, query: str, k: int = 10, candidates: int = 50, rrf_k: int = 60) -> list[int]:
    """Fusion RRF des `candidates` premiers résultats de BM25 et de la recherche dense."""
    return [d for d, _ in reciprocal_rank_fusion(
        [search_bm25(index, query, candidates), search_dense(conn, embedder, query, candidates)], k=rrf_k)][:k]
