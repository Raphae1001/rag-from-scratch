"""Mesure de latence propre (Phase 5) : échauffement puis répétitions, contre l'API `POST /query` en
cours d'exécution (locale ou dans Docker). Contrairement aux latences "indicatives" de la Phase 2
(une seule exécution, pas d'échauffement), ce script sépare explicitement le démarrage à froid des
requêtes en régime établi.

Usage :
    uvicorn rag.api:app &                          # ou : docker compose up -d --wait
    python scripts/measure_latency.py --url http://localhost:8000
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# mélange répondable (retrieval+reranking+génération complets) et hors-corpus (refus plus rapide, pas
# de génération de réponse longue) : les deux profils de latence sont réels, pas la peine de les cacher
QUERIES = [
    "how do I upload a file to the server",
    "how do background tasks work",
    "what is a field validator",
    "how do I run code on startup and shutdown",
    "what is the capital of France",           # hors-corpus : refus attendu
]


def call(url: str, query: str) -> tuple[float, dict]:
    """Renvoie (latence totale HTTP en ms, latency_ms renvoyé par l'API pour cette requête)."""
    payload = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(f"{url}/query", data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    total_ms = (time.perf_counter() - t0) * 1000
    return total_ms, body["latency_ms"]


def stats(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "p50": statistics.median(values),
            "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--warmup", type=int, default=5, help="requêtes d'échauffement avant de mesurer (non comptées)")
    ap.add_argument("--repeats", type=int, default=5, help="répétitions de la liste de requêtes")
    args = ap.parse_args()

    try:
        urllib.request.urlopen(f"{args.url}/health", timeout=5)
    except urllib.error.URLError as e:
        raise SystemExit(f"API injoignable sur {args.url} : {e}. Lancer `uvicorn rag.api:app` ou `docker compose up -d --wait`.")

    print(f"Échauffement : {args.warmup} requêtes...")
    for _ in range(args.warmup):
        call(args.url, QUERIES[0])

    print(f"Mesure : {args.repeats} répétitions x {len(QUERIES)} requêtes = {args.repeats * len(QUERIES)} appels...")
    totals, retrieval, reranking, generation = [], [], [], []
    for _ in range(args.repeats):
        for q in QUERIES:
            total_ms, lat = call(args.url, q)
            totals.append(total_ms)
            retrieval.append(lat["retrieval"])
            reranking.append(lat["reranking"])
            generation.append(lat["generation"])

    rows = {"total (HTTP)": stats(totals), "retrieval": stats(retrieval),
            "reranking": stats(reranking), "génération": stats(generation)}

    md = [f"# Mesure de latence (Phase 5) — {args.warmup} requêtes d'échauffement, "
          f"{args.repeats}x{len(QUERIES)} = {len(totals)} mesures, après échauffement", "",
          "| Étape | moyenne (ms) | p50 (ms) | p95 (ms) |", "|---|---|---|---|"]
    for name, s in rows.items():
        md.append(f"| {name} | {s['mean']:.1f} | {s['p50']:.1f} | {s['p95']:.1f} |")
    md += ["", f"Requêtes utilisées : {', '.join(repr(q) for q in QUERIES)}", "",
           "La génération domine largement (appel réseau à l'API Claude) ; retrieval et reranking sont "
           "des ordres de grandeur plus rapides (mesuré aussi en Phase 3)."]

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "phase5_latency.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "phase5_latency.json").write_text(json.dumps({"warmup": args.warmup, "repeats": args.repeats,
                                                          "n_queries": len(QUERIES), "stats": rows},
                                                         indent=2, default=float), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
