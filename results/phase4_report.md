# Vérification Phase 4 — tableau comparatif final + fidélité (44 requêtes annotées)

## Tableau comparatif final (repris de results/phase2_benchmark.json)

| Configuration | recall@10 | recall@5 | precision@10 | MRR | latence moy. (ms) |
|---|---|---|---|---|---|
| BM25 seul | 0.572 | 0.394 | 0.109 | 0.333 | 0.6 |
| Dense seul | 0.659 | 0.424 | 0.127 | 0.396 | 22.0 |
| Hybride (RRF) | 0.720 | 0.481 | 0.139 | 0.474 | 20.7 |

**Configuration gagnante : Hybride (RRF)** — recall@10 0.720 contre 0.572 pour BM25 seul (+0.148), en combinant la robustesse lexicale de BM25 et la généralisation sémantique du dense sans dégrader la latence (fusion RRF, pas de modèle supplémentaire à ce stade).

## Fidélité (faithfulness) des réponses générées — pipeline Phase 3 complet

41/44 requêtes répondables, 3 refusées, 0 en erreur de format (réponse non-JSON, exclues de la mesure). Score de fidélité moyen sur les répondables : **0.964** (part des affirmations soutenues par les passages cités, jugée par un second appel LLM).

Affirmations jugées NON soutenues par le contexte cité :

- `q01` : The exit code (code after the yield statement) in a dependency with yield is executed after the response is sent to the client
- `q01` : Dependencies with yield are useful for cleanup operations like closing database sessions or releasing resources after the response has been transmitted
- `q02` : `403 Forbidden` is for insufficient privileges.
- `q02` : The `detail` parameter can be any value that can be converted to JSON (string, dict, list, etc.).
- `q12` : FastAPI is compatible with various databases, external APIs, authentication systems, and other components
- `q14` : You need to define your custom exception class.
- `q15` : You can add middleware to FastAPI applications using either the `@app.middleware()` decorator or the `app.add_middleware()` method.
- `q26` : The dependency function should authenticate the user and return user information if the token is valid, or raise an HTTP exception if it's not
- `q26` : You need to use `HTTPBearer` or `HTTPAuthorizationCredentials` specifically
- `q30` : This approach avoids the problem with `PUT` where missing attributes would be replaced with default values
- `q31` : You can define arbitrarily deeply nested models where a model contains lists of other models.
- `q37` : Declare a path parameter with a type annotation using the enum class you created
- `q38` : OAuth2 scopes allow you to verify that all required scopes are included in received tokens
