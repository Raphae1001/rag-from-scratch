# Benchmark Phase 2 (44 requêtes annotées, 1616 documents)

## Retrieval (top-10)

| Configuration | recall@10 | recall@5 | precision@10 | MRR | latence moy. (ms) |
|---|---|---|---|---|---|
| BM25 seul | 0.576 | 0.379 | 0.109 | 0.357 | 0.9 |
| Dense seul | 0.617 | 0.432 | 0.120 | 0.412 | 26.9 |
| Hybride (RRF) | 0.739 | 0.485 | 0.141 | 0.508 | 22.1 |

### recall@10 par type de requête

| Configuration | keyword (n=16) | paraphrase (n=28) |
|---|---|---|
| BM25 seul | 0.844 | 0.423 |
| Dense seul | 0.906 | 0.452 |
| Hybride (RRF) | 1.000 | 0.589 |

## Compromis exactitude / vitesse (recherche approximative HNSW, passages)

Index HNSW m=16, ef_construction=64 : construction 0.78 s.

| Mode | recall@10 vs exact | latence moy. (ms) | p95 (ms) |
|---|---|---|---|
| exact (sans index) | 1.000 | 5.60 | 8.06 |
| HNSW ef_search=10 | 0.957 | 1.02 | 2.33 |
| HNSW ef_search=40 | 0.993 | 1.06 | 2.25 |
| HNSW ef_search=100 | 1.000 | 1.19 | 1.90 |

**Gate Phase 2** — recall@10 hybride (0.739) > BM25 seul (0.576) : **OK**
