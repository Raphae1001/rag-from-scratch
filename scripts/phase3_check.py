"""Vérification des gates de la Phase 3 : amélioration du reranking + 0 hallucination sur le set de non-réponse.

Usage :   python scripts/phase3_check.py                 (résultats -> results/phase3_report.{json,md})
Plomberie hors ligne (reranking seul, pas de génération) : python scripts/phase3_check.py --fake --pgdata data/pgdata
Prérequis : scripts/embed_corpus.py déjà exécuté ; $ANTHROPIC_API_KEY dans l'environnement pour la partie génération.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from _common import ROOT, add_db_args, doc_key, get_dsn, load_corpus

from rag import db
from rag.embed import HashingEmbedder, SentenceTransformerEmbedder
from rag.generate import AnthropicClient, generate_answer
from rag.index import build_index
from rag.metrics import recall_at_k, reciprocal_rank
from rag.rerank import CrossEncoderReranker, OverlapReranker
from rag.retrieval import search_hybrid

K = 10
NO_ANSWER = json.loads((ROOT / "data" / "eval" / "no_answer.json").read_text(encoding="utf-8"))


def rerank_gate(docs, index, conn, embedder, reranker, queries, top_n: int = 30) -> dict:
    """Compare recall@10 / MRR du retrieval hybride seul vs hybride + reranking, sur les requêtes annotées.

    `top_n` : nombre de résultats hybrides (déjà fusionnés) passés au reranker — pas à confondre avec le
    paramètre `candidates` de `search_hybrid`, qui règle un pool interne différent (BM25/dense avant fusion).
    """
    keys = [doc_key(d) for d in docs]
    before, after = [], []
    for q in queries:
        relevant = set(q["relevant"])
        doc_ids = search_hybrid(index, conn, embedder, q["query"], k=top_n)
        before_keys = [keys[i] for i in doc_ids][:K]
        texts = db.get_docs(conn, doc_ids)
        reranked_ids = [d for d, _ in reranker.rerank(q["query"], [(d, texts[d]["text"]) for d in doc_ids if d in texts])]
        after_keys = [keys[i] for i in reranked_ids][:K]
        before.append((recall_at_k(before_keys, relevant, K), reciprocal_rank(before_keys, relevant)))
        after.append((recall_at_k(after_keys, relevant, K), reciprocal_rank(after_keys, relevant)))
    b_recall, b_mrr = np.mean(before, axis=0)
    a_recall, a_mrr = np.mean(after, axis=0)
    return {"n": len(queries), "before": {"recall@10": float(b_recall), "mrr": float(b_mrr)},
            "after": {"recall@10": float(a_recall), "mrr": float(a_mrr)}}


def hallucination_gate(docs, index, conn, embedder, reranker, client, k: int = 5, top_n: int = 20,
                        max_workers: int = 6) -> dict:
    """Sur le set de non-réponse : le système doit répondre "answerable: false" sur 100% des cas.

    Le retrieval + reranking reste séquentiel (partage une seule connexion Postgres, pas thread-safe),
    mais les appels à l'API Claude sont indépendants les uns des autres et dominés par la latence réseau
    (1-3 s chacun) : on les lance en parallèle avec un ThreadPoolExecutor plutôt qu'un par un. Le GIL
    n'est pas un problème ici car ces threads passent l'essentiel de leur temps à attendre le réseau.
    """
    prepared = []
    for q in NO_ANSWER:
        doc_ids = search_hybrid(index, conn, embedder, q["query"], k=top_n)
        texts = db.get_docs(conn, doc_ids)
        reranked = reranker.rerank(q["query"], [(d, texts[d]["text"]) for d in doc_ids if d in texts])[:k]
        passages = [(f"{texts[d]['path']}#{texts[d]['section']}", texts[d]["text"]) for d, _ in reranked]
        prepared.append((q, passages))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        outs = list(pool.map(lambda qp: generate_answer(client, qp[0]["query"], qp[1]), prepared))
    results = [{"id": q["id"], "query": q["query"], "answerable": out["answerable"], "answer": out["answer"]}
               for (q, _), out in zip(prepared, outs)]
    hallucinated = [r for r in results if r["answerable"]]
    return {"n": len(results), "hallucinated": len(hallucinated), "cases": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_db_args(ap)
    ap.add_argument("--fake", action="store_true",
                     help="embeddings + reranker + LLM factices : plomberie seulement, résultats NON valides, pas d'appel réseau")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    docs = load_corpus()
    queries = json.loads((ROOT / "data" / "eval" / "queries.json").read_text(encoding="utf-8"))
    index = build_index([d["text"] for d in docs])

    t0 = time.perf_counter()
    embedder = HashingEmbedder() if args.fake else SentenceTransformerEmbedder(args.model)
    reranker = OverlapReranker() if args.fake else CrossEncoderReranker()
    conn = db.connect(get_dsn(args))
    n_docs, _ = db.counts(conn)
    assert n_docs == len(docs), "embeddings non chargés : lancer scripts/embed_corpus.py (avec les mêmes options)"
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    rerank_result = rerank_gate(docs, index, conn, embedder, reranker, queries)
    t_rerank_gate = time.perf_counter() - t0
    rerank_ok = rerank_result["after"]["recall@10"] > rerank_result["before"]["recall@10"] or \
        rerank_result["after"]["mrr"] > rerank_result["before"]["mrr"]

    warn = ""
    t0 = time.perf_counter()
    if args.fake:
        warn = "\n> ⚠️ PLOMBERIE FACTICE (--fake) : pas de génération réelle, ces chiffres ne mesurent rien. Ne pas les reporter.\n"
        hallu_result = {"n": 0, "hallucinated": 0, "cases": [], "skipped": True}
    else:
        client = AnthropicClient()
        hallu_result = hallucination_gate(docs, index, conn, embedder, reranker, client)
    t_hallu_gate = time.perf_counter() - t0
    hallu_ok = args.fake or hallu_result["hallucinated"] == 0

    print(f"[timing] chargement modèles+DB : {t_load:.1f}s | gate reranking ({len(queries)} requêtes) : "
          f"{t_rerank_gate:.1f}s | gate non-hallucination ({len(NO_ANSWER)} requêtes) : {t_hallu_gate:.1f}s")

    md = [f"# Vérification Phase 3 ({len(queries)} requêtes annotées, {len(NO_ANSWER)} requêtes de non-réponse){warn}",
          "", "## Gate reranking — recall@10 / MRR avant vs après cross-encoder", "",
          "| | recall@10 | MRR |", "|---|---|---|",
          f"| Hybride seul | {rerank_result['before']['recall@10']:.3f} | {rerank_result['before']['mrr']:.3f} |",
          f"| Hybride + reranking | {rerank_result['after']['recall@10']:.3f} | {rerank_result['after']['mrr']:.3f} |",
          "", f"**Gate reranking** — amélioration mesurée : **{'OK' if rerank_ok else 'NON ATTEINT'}**", "",
          "## Gate non-hallucination — set de non-réponse", ""]
    if args.fake:
        md.append("_(sautée en mode --fake, nécessite un vrai appel à l'API Claude)_")
    else:
        md.append(f"{hallu_result['n'] - hallu_result['hallucinated']}/{hallu_result['n']} cas correctement identifiés comme "
                   "« je ne sais pas ».")
        md += ["", "| id | requête | answerable (doit être false) |", "|---|---|---|"]
        md += [f"| {c['id']} | {c['query']} | {c['answerable']} |" for c in hallu_result["cases"]]
    md += ["", f"**Gate non-hallucination** — 0/{len(NO_ANSWER)} halluciné : **{'OK' if hallu_ok else 'NON ATTEINT'}**"]

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    stem = "phase3_report_FAKE" if args.fake else "phase3_report"
    (out / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / f"{stem}.json").write_text(json.dumps({"fake": args.fake, "rerank": rerank_result, "rerank_gate_ok": rerank_ok,
                                                  "hallucination": hallu_result, "hallucination_gate_ok": hallu_ok},
                                                 indent=2, default=float), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
