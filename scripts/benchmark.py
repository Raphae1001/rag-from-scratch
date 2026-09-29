"""Benchmark Phase 2 : BM25 vs dense vs hybride (RRF) sur le set annoté, et compromis exact / HNSW.

Prérequis : scripts/embed_corpus.py exécuté avec le modèle réel.
Usage :   python scripts/benchmark.py            (résultats -> results/phase2_benchmark.{json,md})
Test de plomberie hors ligne : python scripts/benchmark.py --fake --pgdata data/pgdata
"""
import argparse
import json
import statistics
import time

import numpy as np
from _common import ROOT, add_db_args, doc_key, get_dsn, load_corpus, load_eval

from rag import db
from rag.embed import HashingEmbedder, SentenceTransformerEmbedder
from rag.index import build_index
from rag.metrics import precision_at_k, recall_at_k, reciprocal_rank
from rag.retrieval import search_bm25, search_dense, search_hybrid

K = 10


def evaluate(rankings: dict[str, list[str]], queries: list[dict]) -> dict:
    """rankings : id de requête -> clés classées. Renvoie les moyennes globales et par type de requête."""
    def agg(subset):
        rows = [(recall_at_k(rankings[q["id"]], set(q["relevant"]), K),
                 recall_at_k(rankings[q["id"]], set(q["relevant"]), 5),
                 precision_at_k(rankings[q["id"]], set(q["relevant"]), K),
                 reciprocal_rank(rankings[q["id"]], set(q["relevant"]))) for q in subset]
        m = np.mean(rows, axis=0)
        return {"n": len(subset), "recall@10": m[0], "recall@5": m[1], "precision@10": m[2], "mrr": m[3]}
    out = {"all": agg(queries)}
    for kind in sorted({q["kind"] for q in queries}):
        out[kind] = agg([q for q in queries if q["kind"] == kind])
    return out


def ann_tradeoff(conn, qvecs: np.ndarray, repeats: int = 20) -> list[dict]:
    """Recherche exacte vs HNSW au niveau des passages : recall@10 de l'ANN par rapport à l'exact, et latence."""
    sql = "SELECT chunk_id FROM chunks ORDER BY embedding <=> %s LIMIT 10"

    def run(setting_ef=None):
        if setting_ef:
            conn.execute(f"SET hnsw.ef_search = {int(setting_ef)}")
        ids, lat = [], []
        for _ in range(repeats):
            ids = []
            for q in qvecs:
                t0 = time.perf_counter()
                ids.append([r[0] for r in conn.execute(sql, (q,)).fetchall()])
                lat.append((time.perf_counter() - t0) * 1000)
        return ids, statistics.mean(lat), float(np.percentile(lat, 95))

    db.drop_hnsw(conn)
    exact_ids, exact_ms, exact_p95 = run()
    rows = [{"mode": "exact (sans index)", "recall_vs_exact": 1.0, "mean_ms": exact_ms, "p95_ms": exact_p95}]
    t0 = time.perf_counter()
    db.create_hnsw(conn)
    build_s = time.perf_counter() - t0
    conn.execute("ANALYZE chunks")
    for ef in (10, 40, 100):
        ids, ms, p95 = run(ef)
        rec = np.mean([len(set(a) & set(e)) / 10 for a, e in zip(ids, exact_ids)])
        rows.append({"mode": f"HNSW ef_search={ef}", "recall_vs_exact": float(rec), "mean_ms": ms, "p95_ms": p95})
    db.drop_hnsw(conn)
    return rows, build_s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_db_args(ap)
    ap.add_argument("--fake", action="store_true", help="embeddings factices : plomberie seulement, résultats NON valides")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    docs, queries = load_corpus(), load_eval()
    keys = [doc_key(d) for d in docs]
    assert len(set(keys)) == len(keys), "clés de documents non uniques"
    embedder = HashingEmbedder() if args.fake else SentenceTransformerEmbedder(args.model)
    conn = db.connect(get_dsn(args))
    n_docs, _ = db.counts(conn)
    assert n_docs == len(docs), "embeddings non chargés : lancer scripts/embed_corpus.py (avec les mêmes options)"
    index = build_index([d["text"] for d in docs])

    configs = {
        "BM25 seul": lambda q: search_bm25(index, q, K),
        "Dense seul": lambda q: search_dense(conn, embedder, q, K),
        "Hybride (RRF)": lambda q: search_hybrid(index, conn, embedder, q, K),
    }
    results, latencies = {}, {}
    for name, fn in configs.items():
        rankings, times = {}, []
        for q in queries:
            t0 = time.perf_counter()
            rankings[q["id"]] = [keys[i] for i in fn(q["query"])]
            times.append((time.perf_counter() - t0) * 1000)
        results[name], latencies[name] = evaluate(rankings, queries), statistics.mean(times)

    qvecs = embedder.encode([q["query"] for q in queries])
    ann, build_s = ann_tradeoff(conn, qvecs)

    b, h = results["BM25 seul"]["all"]["recall@10"], results["Hybride (RRF)"]["all"]["recall@10"]
    gate = h > b
    warn = "\n> ⚠️ EMBEDDINGS FACTICES : ces chiffres ne mesurent rien de sémantique. Ne pas les reporter.\n" if args.fake else ""
    md = [f"# Benchmark Phase 2 ({len(queries)} requêtes annotées, {len(docs)} documents){warn}", "",
          "## Retrieval (top-10)", "", "| Configuration | recall@10 | recall@5 | precision@10 | MRR | latence moy. (ms) |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        a = r["all"]
        md.append(f"| {name} | {a['recall@10']:.3f} | {a['recall@5']:.3f} | {a['precision@10']:.3f} | {a['mrr']:.3f} | {latencies[name]:.1f} |")
    md += ["", "### recall@10 par type de requête", "", "| Configuration | " + " | ".join(f"{k} (n={results['BM25 seul'][k]['n']})" for k in results["BM25 seul"] if k != "all") + " |",
           "|---|" + "---|" * (len(results["BM25 seul"]) - 1)]
    for name, r in results.items():
        md.append(f"| {name} | " + " | ".join(f"{r[k]['recall@10']:.3f}" for k in r if k != "all") + " |")
    md += ["", "## Compromis exactitude / vitesse (recherche approximative HNSW, passages)", "",
           f"Index HNSW m=16, ef_construction=64 : construction {build_s:.2f} s.", "",
           "| Mode | recall@10 vs exact | latence moy. (ms) | p95 (ms) |", "|---|---|---|---|"]
    md += [f"| {r['mode']} | {r['recall_vs_exact']:.3f} | {r['mean_ms']:.2f} | {r['p95_ms']:.2f} |" for r in ann]
    md += ["", f"**Gate Phase 2** — recall@10 hybride ({h:.3f}) > BM25 seul ({b:.3f}) : **{'OK' if gate else 'NON ATTEINT'}**"]

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    stem = "phase2_benchmark_FAKE" if args.fake else "phase2_benchmark"
    (out / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / f"{stem}.json").write_text(json.dumps({"fake_embeddings": args.fake, "model": None if args.fake else args.model,
                                                  "n_queries": len(queries), "retrieval": results, "latency_ms": latencies,
                                                  "ann": ann, "hnsw_build_s": build_s, "gate_hybrid_beats_bm25": gate},
                                                 indent=2, default=float), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
