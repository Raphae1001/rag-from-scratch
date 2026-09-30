"""Intervalle de confiance bootstrap sur le gain recall@10 hybride vs BM25 (Phase 2).

Le README affichait un écart net (0,739 vs 0,576) sans intervalle de confiance — juste "44 requêtes,
pas de quoi calculer une marge d'erreur". Ce script comble ça : on rééchantillonne les 44 requêtes avec
remise (bootstrap non paramétrique), on recalcule le recall@10 moyen de chaque configuration à chaque
tirage, et on prend les percentiles 2,5/97,5 de la distribution obtenue comme IC à 95%.

Usage : python scripts/bootstrap_ci.py [--n-resamples 10000]
"""
import argparse
import json

import numpy as np
from _common import ROOT, add_db_args, doc_key, get_dsn, load_corpus, load_eval

from rag import db
from rag.embed import SentenceTransformerEmbedder
from rag.index import build_index
from rag.metrics import recall_at_k
from rag.retrieval import search_bm25, search_hybrid

K = 10


def per_query_recall(index, conn, embedder, docs, queries) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (recall@10 BM25, recall@10 hybride) par requête, alignés sur `queries`."""
    keys = [doc_key(d) for d in docs]
    bm25, hybrid = [], []
    for q in queries:
        relevant = set(q["relevant"])
        bm25.append(recall_at_k([keys[i] for i in search_bm25(index, q["query"], k=K)], relevant, K))
        hybrid.append(recall_at_k([keys[i] for i in search_hybrid(index, conn, embedder, q["query"], k=K)], relevant, K))
    return np.array(bm25), np.array(hybrid)


def bootstrap_ci(values: np.ndarray, n_resamples: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """(moyenne observée, borne basse IC95%, borne haute IC95%) par rééchantillonnage avec remise."""
    n = len(values)
    means = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_resamples)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_db_args(ap)
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0, help="graine du générateur aléatoire (reproductibilité)")
    args = ap.parse_args()

    docs, queries = load_corpus(), load_eval()
    index = build_index([d["text"] for d in docs])
    embedder = SentenceTransformerEmbedder()
    conn = db.connect(get_dsn(args))
    n_docs, _ = db.counts(conn)
    assert n_docs == len(docs), "embeddings non chargés : lancer scripts/embed_corpus.py"

    bm25, hybrid = per_query_recall(index, conn, embedder, docs, queries)
    diff = hybrid - bm25
    rng = np.random.default_rng(args.seed)

    bm25_mean, bm25_lo, bm25_hi = bootstrap_ci(bm25, args.n_resamples, rng)
    hyb_mean, hyb_lo, hyb_hi = bootstrap_ci(hybrid, args.n_resamples, rng)
    diff_mean, diff_lo, diff_hi = bootstrap_ci(diff, args.n_resamples, rng)
    gate_significant = diff_lo > 0  # l'IC95% de l'écart exclut 0 -> gain statistiquement significatif à ce niveau

    md = [f"# Intervalle de confiance bootstrap — recall@10 hybride vs BM25 ({len(queries)} requêtes, "
          f"{args.n_resamples} rééchantillonnages)", "",
          "| | moyenne | IC 95% |", "|---|---|---|",
          f"| BM25 seul | {bm25_mean:.3f} | [{bm25_lo:.3f}, {bm25_hi:.3f}] |",
          f"| Hybride (RRF) | {hyb_mean:.3f} | [{hyb_lo:.3f}, {hyb_hi:.3f}] |",
          f"| **Écart (hybride − BM25)** | **{diff_mean:.3f}** | **[{diff_lo:.3f}, {diff_hi:.3f}]** |", "",
          f"**L'IC95% de l'écart exclut 0 : {'oui' if gate_significant else 'non'}** — le gain de l'hybride sur "
          f"BM25 est {'statistiquement significatif' if gate_significant else 'PAS statistiquement significatif'} "
          f"à ce niveau de confiance, sur ces 44 requêtes.", "",
          "Méthode : bootstrap non paramétrique — 44 requêtes rééchantillonnées avec remise à chaque tirage, "
          "recall@10 moyen recalculé, IC = percentiles 2,5/97,5 de la distribution des moyennes rééchantillonnées. "
          "Ne corrige pas le biais d'échantillonnage des annotations elles-mêmes (voir limites du set d'éval, "
          "README section Phase 2) — seulement l'incertitude due à la petite taille de l'échantillon (n=44)."]

    out = ROOT / "results"
    (out / "phase2_bootstrap_ci.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "phase2_bootstrap_ci.json").write_text(json.dumps({
        "n_queries": len(queries), "n_resamples": args.n_resamples, "seed": args.seed,
        "bm25": {"mean": bm25_mean, "ci95": [bm25_lo, bm25_hi]},
        "hybrid": {"mean": hyb_mean, "ci95": [hyb_lo, hyb_hi]},
        "diff": {"mean": diff_mean, "ci95": [diff_lo, diff_hi]},
        "gate_significant_at_95": gate_significant,
    }, indent=2, default=float), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
