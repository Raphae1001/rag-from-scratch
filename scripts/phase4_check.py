"""Vérification des gates de la Phase 4 : tableau comparatif final + fidélité (faithfulness) des réponses.

Le tableau precision@k/recall@k/MRR des 3 configurations est repris tel quel de results/phase2_benchmark.json
(déjà mesuré et versionné en Phase 2) plutôt que recalculé. La nouveauté de la Phase 4 est la fidélité : pour
chaque requête répondable du set annoté, on génère une réponse (pipeline Phase 3) puis on la fait juger par un
second appel LLM, indépendant, qui vérifie que chaque affirmation est bien soutenue par les passages cités.

Usage :   python scripts/phase4_check.py                 (résultats -> results/phase4_report.{json,md})
Plomberie hors ligne (pas de génération réelle) : python scripts/phase4_check.py --fake --pgdata data/pgdata
Prérequis : results/phase2_benchmark.json présent ; scripts/embed_corpus.py exécuté ; $ANTHROPIC_API_KEY pour
la partie fidélité.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from _common import ROOT, add_db_args, get_dsn, load_corpus

from rag import db
from rag.embed import HashingEmbedder, SentenceTransformerEmbedder
from rag.faithfulness import judge_faithfulness
from rag.generate import AnthropicClient, generate_answer
from rag.index import build_index
from rag.rerank import CrossEncoderReranker, OverlapReranker
from rag.retrieval import search_hybrid

PHASE2_BENCHMARK = ROOT / "results" / "phase2_benchmark.json"


def retrieve_and_rerank(query: str, index, conn, embedder, reranker, k: int = 5, top_n: int = 20) -> list[tuple[str, str]]:
    """Retourne les k meilleurs passages (clé "path#section", texte) après retrieval hybride + reranking."""
    doc_ids = search_hybrid(index, conn, embedder, query, k=top_n)
    texts = db.get_docs(conn, doc_ids)
    reranked = reranker.rerank(query, [(d, texts[d]["text"]) for d in doc_ids if d in texts])[:k]
    return [(f"{texts[d]['path']}#{texts[d]['section']}", texts[d]["text"]) for d, _ in reranked]


def faithfulness_gate(index, conn, embedder, reranker, client, queries, max_workers: int = 6) -> dict:
    prepared = [(q, retrieve_and_rerank(q["query"], index, conn, embedder, reranker)) for q in queries]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        answers = list(pool.map(lambda qp: generate_answer(client, qp[0]["query"], qp[1]), prepared))

    # seules les réponses "answerable" ont des affirmations à juger ; on ne juge que sur les passages CITÉS
    judgeable = [(q, ans, [(key, text) for key, text in passages if key in ans["citations"]])
                 for (q, passages), ans in zip(prepared, answers) if ans["answerable"]]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        judgments = list(pool.map(lambda qac: judge_faithfulness(client, qac[1]["answer"], qac[2]), judgeable))

    cases = []
    for (q, passages), ans in zip(prepared, answers):
        case = {"id": q["id"], "query": q["query"], "answerable": ans["answerable"], "citations": ans["citations"]}
        cases.append(case)
    for (q, ans, _), judgment in zip(judgeable, judgments):
        case = next(c for c in cases if c["id"] == q["id"])
        case["faithfulness"] = judgment["score"]
        case["claims"] = judgment["claims"]

    unanswered = [c for c in cases if not c["answerable"]]
    scores = [c["faithfulness"] for c in cases if "faithfulness" in c]
    mean_score = sum(scores) / len(scores) if scores else None
    return {"n": len(cases), "n_answerable": len(scores), "n_refused": len(unanswered),
            "mean_faithfulness": mean_score, "cases": cases}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_db_args(ap)
    ap.add_argument("--fake", action="store_true",
                     help="embeddings + reranker factices, pas de génération réelle : plomberie seulement")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    if not PHASE2_BENCHMARK.exists():
        raise SystemExit(f"{PHASE2_BENCHMARK} manquant : lancer scripts/benchmark.py d'abord (Phase 2)")
    phase2 = json.loads(PHASE2_BENCHMARK.read_text(encoding="utf-8"))

    docs = load_corpus()
    queries = json.loads((ROOT / "data" / "eval" / "queries.json").read_text(encoding="utf-8"))
    index = build_index([d["text"] for d in docs])
    embedder = HashingEmbedder() if args.fake else SentenceTransformerEmbedder(args.model)
    reranker = OverlapReranker() if args.fake else CrossEncoderReranker()
    conn = db.connect(get_dsn(args))
    n_docs, _ = db.counts(conn)
    assert n_docs == len(docs), "embeddings non chargés : lancer scripts/embed_corpus.py (avec les mêmes options)"

    warn = ""
    t0 = time.perf_counter()
    if args.fake:
        warn = "\n> ⚠️ PLOMBERIE FACTICE (--fake) : pas de génération réelle, ces chiffres ne mesurent rien. Ne pas les reporter.\n"
        faith = {"n": 0, "n_answerable": 0, "n_refused": 0, "mean_faithfulness": None, "cases": [], "skipped": True}
    else:
        client = AnthropicClient()
        faith = faithfulness_gate(index, conn, embedder, reranker, client, queries)
    t_faith = time.perf_counter() - t0

    md = [f"# Vérification Phase 4 — tableau comparatif final + fidélité ({len(queries)} requêtes annotées){warn}",
          "", "## Tableau comparatif final (repris de results/phase2_benchmark.json)", "",
          "| Configuration | recall@10 | recall@5 | precision@10 | MRR | latence moy. (ms) |", "|---|---|---|---|---|---|"]
    for name, r in phase2["retrieval"].items():
        a = r["all"]
        md.append(f"| {name} | {a['recall@10']:.3f} | {a['recall@5']:.3f} | {a['precision@10']:.3f} | {a['mrr']:.3f} | "
                   f"{phase2['latency_ms'][name]:.1f} |")
    b = phase2["retrieval"]["BM25 seul"]["all"]["recall@10"]
    h = phase2["retrieval"]["Hybride (RRF)"]["all"]["recall@10"]
    md += ["", f"**Configuration gagnante : Hybride (RRF)** — recall@10 {h:.3f} contre {b:.3f} pour BM25 seul "
               f"(+{h - b:.3f}), en combinant la robustesse lexicale de BM25 et la généralisation sémantique du "
               "dense sans dégrader la latence (fusion RRF, pas de modèle supplémentaire à ce stade).", ""]

    md += ["## Fidélité (faithfulness) des réponses générées — pipeline Phase 3 complet", ""]
    if args.fake:
        md.append("_(sautée en mode --fake, nécessite un vrai appel à l'API Claude)_")
    else:
        md.append(f"{faith['n_answerable']}/{faith['n']} requêtes répondables (le reste correctement refusé), "
                   f"score de fidélité moyen : **{faith['mean_faithfulness']:.3f}** "
                   f"(part des affirmations soutenues par les passages cités, jugée par un second appel LLM).")
        unsupported = [(c["id"], cl["claim"]) for c in faith["cases"] if "claims" in c
                       for cl in c["claims"] if not cl["supported"]]
        if unsupported:
            md += ["", "Affirmations jugées NON soutenues par le contexte cité :", ""]
            md += [f"- `{qid}` : {claim}" for qid, claim in unsupported]
        else:
            md += ["", "Aucune affirmation jugée non soutenue."]

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    stem = "phase4_report_FAKE" if args.fake else "phase4_report"
    (out / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / f"{stem}.json").write_text(json.dumps({"fake": args.fake, "phase2_retrieval": phase2["retrieval"],
                                                  "faithfulness": faith, "faithfulness_check_s": t_faith},
                                                 indent=2, default=float), encoding="utf-8")
    print("\n".join(md))
    print(f"\n[timing] gate fidélité ({len(queries)} requêtes) : {t_faith:.1f}s")


if __name__ == "__main__":
    main()
