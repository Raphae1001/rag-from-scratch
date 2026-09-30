# Mesure de latence (Phase 5) — 3 requêtes d'échauffement, 3x5 = 15 mesures, après échauffement

| Étape | moyenne (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|
| total (HTTP) | 3712.2 | 3924.6 | 4751.5 |
| retrieval | 39.7 | 40.3 | 52.9 |
| reranking | 975.3 | 968.9 | 1087.9 |
| génération | 2557.9 | 2773.3 | 3619.6 |

Requêtes utilisées : 'how do I upload a file to the server', 'how do background tasks work', 'what is a field validator', 'how do I run code on startup and shutdown', 'what is the capital of France'

La génération domine largement (appel réseau à l'API Claude) ; retrieval et reranking sont des ordres de grandeur plus rapides (mesuré aussi en Phase 3).
