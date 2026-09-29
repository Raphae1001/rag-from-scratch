# Benchmark Phase 2 (44 requêtes annotées, 1597 documents)

## Retrieval (top-10)

| Configuration | recall@10 | recall@5 | precision@10 | MRR | latence moy. (ms) |
|---|---|---|---|---|---|
| BM25 seul | 0.572 | 0.394 | 0.109 | 0.333 | 0.6 |
| Dense seul | 0.659 | 0.424 | 0.127 | 0.396 | 108.7 |
| Hybride (RRF) | 0.720 | 0.481 | 0.139 | 0.474 | 21.0 |

### recall@10 par type de requête

| Configuration | keyword (n=16) | paraphrase (n=28) |
|---|---|---|
| BM25 seul | 0.812 | 0.435 |
| Dense seul | 0.938 | 0.500 |
| Hybride (RRF) | 0.938 | 0.595 |

## Compromis exactitude / vitesse (recherche approximative HNSW, passages)

Index HNSW m=16, ef_construction=64 : construction 0.52 s.

| Mode | recall@10 vs exact | latence moy. (ms) | p95 (ms) |
|---|---|---|---|
| exact (sans index) | 1.000 | 3.72 | 5.26 |
| HNSW ef_search=10 | 0.941 | 0.64 | 0.91 |
| HNSW ef_search=40 | 0.993 | 0.73 | 0.90 |
| HNSW ef_search=100 | 0.998 | 0.93 | 1.20 |

**Gate Phase 2** — recall@10 hybride (0.720) > BM25 seul (0.572) : **OK**
