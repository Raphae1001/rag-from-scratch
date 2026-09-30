# Vérification Phase 4 — tableau comparatif final + fidélité (44 requêtes annotées)

## Tableau comparatif final (repris de results/phase2_benchmark.json)

| Configuration | recall@10 | recall@5 | precision@10 | MRR | latence moy. (ms) |
|---|---|---|---|---|---|
| BM25 seul | 0.576 | 0.379 | 0.109 | 0.357 | 0.9 |
| Dense seul | 0.617 | 0.432 | 0.120 | 0.412 | 26.9 |
| Hybride (RRF) | 0.739 | 0.485 | 0.141 | 0.508 | 22.1 |

**Configuration gagnante : Hybride (RRF)** — recall@10 0.739 contre 0.576 pour BM25 seul (+0.163), en combinant la robustesse lexicale de BM25 et la généralisation sémantique du dense sans dégrader la latence (fusion RRF, pas de modèle supplémentaire à ce stade).

## Fidélité (faithfulness) des réponses générées — pipeline Phase 3 complet

42/44 requêtes répondables, 2 refusées, 0 en erreur de format (réponse non-JSON, exclues de la mesure). Score de fidélité moyen sur les répondables : **0.949** (part des affirmations soutenues par les passages cités, jugée par un second appel LLM).

Affirmations jugées NON soutenues par le contexte cité :

- `q01` : Code after the `yield` statement in a dependency will execute after the response is sent to the client
- `q01` : Dependencies with `yield` are useful for cleanup operations like closing database sessions or releasing resources
- `q01` : After the response is sent to the client, background tasks are executed, and then any cleanup code in dependencies with `yield` runs afterward
- `q02` : You can pass any JSON-convertible value (string, dict, list, etc.) as the detail parameter
- `q02` : 401 is used for authentication failures
- `q02` : 403 is used for permission issues
- `q04` : Using `['*']` is less secure.
- `q12` : FastAPI's dependency injection system is compatible with relational databases
- `q12` : FastAPI's dependency injection system is compatible with NoSQL databases
- `q12` : FastAPI's dependency injection system is compatible with external packages and APIs
- `q13` : Field after validators validate after the field value is parsed.
- `q13` : Field before validators validate before the field value is parsed.
- `q13` : Field plain validators are a plain validation mode.
- `q13` : Field wrap validators are a wrap validation mode.
- `q26` : You can then validate the token contents further in your endpoint or in a separate dependency function (like `get_current_user`) to ensure the token is legitimate before allowing access
- `q27` : `async def` is preferred for better performance in most cases
- `q27` : `async def` is especially preferred for compute-only operations or non-blocking code
- `q35` : You can override FastAPI's default configuration parameters by setting different values in the `swagger_ui_parameters` dictionary
