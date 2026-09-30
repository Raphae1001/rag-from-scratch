"""Pipeline complet (retrieval hybride -> reranking -> génération sourcée), partagé par le CLI
(`scripts/answer.py`) et l'API (Phase 5). Mesure la latence de chaque étape : c'est ce que
`scripts/measure_latency.py` agrège pour la mesure propre (échauffement + répétitions) de la Phase 5.
"""
import time

from rag import db
from rag.generate import LLMClient, generate_answer
from rag.rerank import Reranker
from rag.retrieval import search_hybrid


def answer(query: str, conn, index, embedder, reranker: Reranker, client: LLMClient,
           k: int = 5, top_n: int = 20) -> dict:
    """`top_n` : nombre de résultats hybrides (déjà fusionnés) passés au reranker — pas à confondre avec le
    paramètre `candidates` de `search_hybrid`, qui règle un pool interne différent (BM25/dense avant fusion).

    Renvoie les clés de `generate_answer` (`answerable`, `answer`, `citations`) plus `latency_ms`, un dict
    {"retrieval": ..., "reranking": ..., "generation": ...} en millisecondes.
    """
    t0 = time.perf_counter()
    doc_ids = search_hybrid(index, conn, embedder, query, k=top_n)
    docs = db.get_docs(conn, doc_ids)
    t_retrieval = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    reranked = reranker.rerank(query, [(d, docs[d]["text"]) for d in doc_ids if d in docs])[:k]
    t_reranking = (time.perf_counter() - t0) * 1000

    passages = [(f"{docs[d]['path']}#{docs[d]['section']}", docs[d]["text"]) for d, _ in reranked]

    t0 = time.perf_counter()
    result = generate_answer(client, query, passages)
    t_generation = (time.perf_counter() - t0) * 1000

    return {**result, "latency_ms": {"retrieval": t_retrieval, "reranking": t_reranking, "generation": t_generation}}
