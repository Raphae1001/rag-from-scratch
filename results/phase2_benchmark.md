# Benchmark Phase 2 (44 requêtes annotées, 1597 documents)

## Retrieval (top-10)

| Configuration | recall@10 | recall@5 | precision@10 | MRR | latence moy. (ms) |
|---|---|---|---|---|---|
| BM25 seul | 0.572 | 0.394 | 0.109 | 0.333 | 0.6 |
| Dense seul | 0.659 | 0.424 | 0.127 | 0.396 | 22.0 |
| Hybride (RRF) | 0.720 | 0.481 | 0.139 | 0.474 | 20.7 |

### recall@10 par type de requête

| Configuration | keyword (n=16) | paraphrase (n=28) |
|---|---|---|
| BM25 seul | 0.812 | 0.435 |
| Dense seul | 0.938 | 0.500 |
| Hybride (RRF) | 0.938 | 0.595 |

## Compromis exactitude / vitesse (recherche approximative HNSW, passages)

Index HNSW m=16, ef_construction=64 : construction 0.50 s.

| Mode | recall@10 vs exact | latence moy. (ms) | p95 (ms) |
|---|---|---|---|
| exact (sans index) | 1.000 | 3.44 | 3.76 |
| HNSW ef_search=10 | 0.945 | 0.59 | 0.70 |
| HNSW ef_search=40 | 0.989 | 0.70 | 0.85 |
| HNSW ef_search=100 | 0.998 | 0.87 | 1.04 |

**Gate Phase 2** — recall@10 hybride (0.720) > BM25 seul (0.572) : **OK**
