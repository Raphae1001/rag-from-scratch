"""Ré-évaluation complète (style Phase 4) avec le modèle d'embeddings fine-tuné (Phase 6).

Recherche dense et hybride recalculées **en mémoire** (numpy, pas Postgres) : le modèle fine-tuné produit
des vecteurs différents du modèle de base, donc on ne peut pas réutiliser la table `chunks` existante (qui
contient les embeddings du modèle de base) sans un second passage d'indexation. Reproduire le calcul en
mémoire pour ~2750 passages est simple et rapide, et évite de toucher la base `rag` pour une comparaison.

Usage : python scripts/phase6_check.py [--model models/finetuned-minilm-lora]
"""
import argparse
import json

import numpy as np
from _common import ROOT, doc_key, load_corpus, load_eval

from rag.bm25 import search as bm25_search
from rag.chunking import make_chunks
from rag.embed import SentenceTransformerEmbedder
from rag.fusion import reciprocal_rank_fusion
from rag.index import build_index
from rag.metrics import precision_at_k, recall_at_k, reciprocal_rank

K = 10
PHASE2_BENCHMARK = ROOT / "results" / "phase2_benchmark.json"


def embed_corpus_in_memory(docs: list[dict], embedder) -> tuple[list[int], np.ndarray]:
    """Découpe et embedde tout le corpus ; renvoie (doc_id par passage, matrice des vecteurs normalisés)."""
    doc_ids, texts = [], []
    for d in docs:
        for piece in make_chunks(d):
            doc_ids.append(d["id"])
            texts.append(piece)
    vectors = embedder.encode(texts)
    return doc_ids, vectors


def dense_search_in_memory(query_vec: np.ndarray, chunk_doc_ids: list[int], chunk_vectors: np.ndarray, k: int) -> list[int]:
    """Meilleur passage par document (comme db.dense_search), sans Postgres : produit scalaire = cosinus
    puisque les vecteurs sont normalisés. Corpus assez petit (~2750 passages) pour parcourir tous les
    scores sans index approximatif."""
    sims = chunk_vectors @ query_vec
    best: dict[int, float] = {}
    for i in np.argsort(-sims):  # décroissant : le premier score vu pour un doc_id est son meilleur passage
        doc_id = chunk_doc_ids[i]
        if doc_id not in best:
            best[doc_id] = sims[i]
    return [d for d, _ in sorted(best.items(), key=lambda kv: -kv[1])[:k]]


def evaluate(index, chunk_doc_ids, chunk_vectors, embedder, docs, queries) -> dict:
    keys = [doc_key(d) for d in docs]
    rows_dense, rows_hybrid = [], []
    query_vecs = embedder.encode([q["query"] for q in queries])
    for q, qvec in zip(queries, query_vecs):
        relevant = set(q["relevant"])
        bm25_ids = [d for d, _ in bm25_search(index, q["query"], k=50)]
        dense_ids = dense_search_in_memory(qvec, chunk_doc_ids, chunk_vectors, k=50)
        dense_top10 = [keys[i] for i in dense_ids[:K]]
        hybrid_ids = [d for d, _ in reciprocal_rank_fusion([bm25_ids, dense_ids], k=60)][:K]
        hybrid_top10 = [keys[i] for i in hybrid_ids]
        rows_dense.append((recall_at_k(dense_top10, relevant, K), precision_at_k(dense_top10, relevant, K),
                            reciprocal_rank(dense_top10, relevant)))
        rows_hybrid.append((recall_at_k(hybrid_top10, relevant, K), precision_at_k(hybrid_top10, relevant, K),
                             reciprocal_rank(hybrid_top10, relevant)))
    d, h = np.mean(rows_dense, axis=0), np.mean(rows_hybrid, axis=0)
    return {"dense": {"recall@10": float(d[0]), "precision@10": float(d[1]), "mrr": float(d[2])},
            "hybrid": {"recall@10": float(h[0]), "precision@10": float(h[1]), "mrr": float(h[2])}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "finetuned-minilm-lora"))
    args = ap.parse_args()

    if not PHASE2_BENCHMARK.exists():
        raise SystemExit(f"{PHASE2_BENCHMARK} manquant : lancer scripts/benchmark.py d'abord (Phase 2)")
    phase2 = json.loads(PHASE2_BENCHMARK.read_text(encoding="utf-8"))
    baseline_dense = phase2["retrieval"]["Dense seul"]["all"]
    baseline_hybrid = phase2["retrieval"]["Hybride (RRF)"]["all"]

    docs, queries = load_corpus(), load_eval()
    index = build_index([d["text"] for d in docs])
    embedder = SentenceTransformerEmbedder(args.model)
    chunk_doc_ids, chunk_vectors = embed_corpus_in_memory(docs, embedder)

    result = evaluate(index, chunk_doc_ids, chunk_vectors, embedder, docs, queries)
    d, h = result["dense"], result["hybrid"]
    delta_dense = d["recall@10"] - baseline_dense["recall@10"]
    delta_hybrid = h["recall@10"] - baseline_hybrid["recall@10"]
    gate_ok = delta_dense > 0 or delta_hybrid > 0

    md = [f"# Vérification Phase 6 — fine-tuning contrastif ({len(queries)} requêtes annotées, "
          f"{len(chunk_doc_ids)} passages)", "",
          "## recall@10 : modèle de base (Phase 2) vs modèle fine-tuné", "",
          "| | recall@10 | precision@10 | MRR |", "|---|---|---|---|",
          f"| Dense (base, all-MiniLM-L6-v2) | {baseline_dense['recall@10']:.3f} | {baseline_dense['precision@10']:.3f} | {baseline_dense['mrr']:.3f} |",
          f"| **Dense (fine-tuné LoRA)** | **{d['recall@10']:.3f}** | {d['precision@10']:.3f} | {d['mrr']:.3f} |",
          f"| Hybride (base) | {baseline_hybrid['recall@10']:.3f} | {baseline_hybrid['precision@10']:.3f} | {baseline_hybrid['mrr']:.3f} |",
          f"| **Hybride (fine-tuné)** | **{h['recall@10']:.3f}** | {h['precision@10']:.3f} | {h['mrr']:.3f} |", "",
          f"**Δ recall@10 dense** : {delta_dense:+.3f} | **Δ recall@10 hybride** : {delta_hybrid:+.3f}", "",
          f"**Gate Phase 6** — amélioration mesurée (dense ou hybride) : **{'OK' if gate_ok else 'NON ATTEINT'}**"]

    out = ROOT / "results"
    (out / "phase6_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "phase6_report.json").write_text(json.dumps({
        "baseline": {"dense": baseline_dense, "hybrid": baseline_hybrid}, "finetuned": result,
        "delta_dense_recall@10": delta_dense, "delta_hybrid_recall@10": delta_hybrid, "gate_ok": gate_ok,
    }, indent=2, default=float), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
