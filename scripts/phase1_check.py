"""Vérification Phase 1 : rebuild complet de l'index, mesures, et 5 requêtes de contrôle.

Usage: python scripts/phase1_check.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag.bm25 import search  # noqa: E402
from rag.index import build_index  # noqa: E402

QUERIES = [
    "dependency injection",
    "validate request body",
    "custom exception handler",
    "background tasks",
    "field validator",
]


def main():
    docs = [json.loads(l) for l in (ROOT / "data" / "corpus.jsonl").open(encoding="utf-8")]
    texts = [d["text"] for d in docs]

    t0 = time.perf_counter()
    idx = build_index(texts)
    build_s = time.perf_counter() - t0
    n_post = sum(len(p) for p in idx.postings.values())
    print(f"INDEX: {idx.n_docs} docs | {len(idx.postings)} termes | {n_post} postings | "
          f"avgdl={idx.avgdl:.1f} | build={build_s*1000:.0f} ms\n")

    lat = []
    for q in QUERIES:
        t0 = time.perf_counter()
        res = search(idx, q, k=5)
        lat.append(time.perf_counter() - t0)
        print(f"=== {q!r}")
        for rank, (doc_id, s) in enumerate(res, 1):
            d = docs[doc_id]
            print(f"  {rank}. {s:6.2f}  [{d['source']}] {d['path'].split('/', 1)[1]} > {d['section']}")
        print()
    print(f"latence moyenne par requête: {sum(lat)/len(lat)*1000:.2f} ms")


if __name__ == "__main__":
    main()
