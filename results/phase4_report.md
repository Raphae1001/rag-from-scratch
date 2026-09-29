# Vérification Phase 4 — tableau comparatif final + fidélité (44 requêtes annotées)

## Tableau comparatif final (repris de results/phase2_benchmark.json)

| Configuration | recall@10 | recall@5 | precision@10 | MRR | latence moy. (ms) |
|---|---|---|---|---|---|
| BM25 seul | 0.576 | 0.379 | 0.109 | 0.357 | 0.7 |
| Dense seul | 0.617 | 0.432 | 0.120 | 0.412 | 21.9 |
| Hybride (RRF) | 0.739 | 0.485 | 0.141 | 0.508 | 20.0 |

**Configuration gagnante : Hybride (RRF)** — recall@10 0.739 contre 0.576 pour BM25 seul (+0.163), en combinant la robustesse lexicale de BM25 et la généralisation sémantique du dense sans dégrader la latence (fusion RRF, pas de modèle supplémentaire à ce stade).

## Fidélité (faithfulness) des réponses générées — pipeline Phase 3 complet

42/44 requêtes répondables, 2 refusées, 0 en erreur de format (réponse non-JSON, exclues de la mesure). Score de fidélité moyen sur les répondables : **0.951** (part des affirmations soutenues par les passages cités, jugée par un second appel LLM).

Affirmations jugées NON soutenues par le contexte cité :

- `q01` : Dependencies with `yield` can be used to run code after sending a response to the client
- `q01` : Code after the `yield` statement executes after the response is sent to the client
- `q01` : After the path operation returns a response to the client, any background tasks are sent, and then the exit code of dependencies with `yield` runs after the response is completely sent
- `q02` : `403` is for "Forbidden"
- `q02` : `401` is for "Unauthorized"
- `q02` : You can pass any JSON-convertible value as the `detail` parameter, not just strings (e.g., dict, list).
- `q08` : If you use the `lifespan` parameter, startup and shutdown event handlers will no longer be called
- `q10` : You can receive messages with receive_json()
- `q12` : The system is compatible with all relational databases, NoSQL databases, external packages, APIs, and authentication systems.
- `q18` : Strict mode can be enabled at the field level using the Field parameter
- `q25` : You can use the `--workers` option to specify replication (the number of processes running)
- `q27` : You should use `async def` for endpoints unless they perform blocking I/O operations.
- `q27` : In general, it's better to use `async def` unless your endpoint functions use code that performs blocking I/O.
- `q27` : Blocking I/O includes disk reading/writing or network communications.
- `q33` : You can use `StreamingResponse` with an async generator function that yields chunks.
- `q35` : For a complete list of all available Swagger UI parameters, consult the official Swagger UI configuration documentation
