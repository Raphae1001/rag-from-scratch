"""Source d'écriture du set d'évaluation : résout les préfixes de sections contre le corpus
et écrit data/eval/queries.json (clés exactes "chemin#section").

Usage: python scripts/make_eval_set.py
Chaque pertinent = (chemin de page, préfixe de titre de section) ; le préfixe doit désigner
exactement une section de la page, sinon le script échoue (pas d'annotation approximative).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K, P = "keyword", "paraphrase"
F, S, PY = "fastapi/", "starlette/", "pydantic/"

# (id, kind, requête, [(page, préfixe de section), ...], réponse attendue)
RAW = [
    ("q01", P, "how to run code after sending the response to the client",
     [(F + "tutorial/background-tasks.md", "Using `BackgroundTasks`"), (F + "tutorial/background-tasks.md", "Add the background task")],
     "Declare a `BackgroundTasks` parameter in the path operation function and call `add_task` with the function to run; it runs after the response is sent."),
    ("q02", P, "return an error to the client with a specific status code",
     [(F + "tutorial/handling-errors.md", "Raise an `HTTPException` in your code"), (F + "tutorial/handling-errors.md", "The resulting response")],
     "Raise `HTTPException(status_code=..., detail=...)`; FastAPI returns a JSON response with that status code."),
    ("q03", P, "change how validation errors are returned",
     [(F + "tutorial/handling-errors.md", "Override request validation exceptions"), (F + "tutorial/handling-errors.md", "Override the default exception handler")],
     "Register an exception handler for `RequestValidationError` with `@app.exception_handler` to customize the response."),
    ("q04", P, "allow requests from a frontend running on a different domain",
     [(F + "tutorial/cors.md", "Use `CORSMiddleware`"), (S + "middleware.md", "CORSMiddleware")],
     "Add `CORSMiddleware` with the allowed origins, methods and headers."),
    ("q05", P, "share one database session across all my endpoints",
     [(F + "tutorial/dependencies/dependencies-with-yield.md", "A database dependency with `yield`"), (F + "tutorial/sql-databases.md", "Create a Session Dependency")],
     "Write a dependency that opens a session, yields it, and closes it afterwards, then inject it with `Depends`."),
    ("q06", P, "upload a file to the server",
     [(F + "tutorial/request-files.md", "Define `File` Parameters"), (F + "tutorial/request-files.md", "File Parameters with `UploadFile`")],
     "Declare a parameter of type `UploadFile` (or `bytes` with `File`) in the path operation function."),
    ("q07", P, "hash passwords and generate access tokens",
     [(F + "tutorial/security/oauth2-jwt.md", "Password hashing"), (F + "tutorial/security/oauth2-jwt.md", "Hash and verify the passwords"), (F + "tutorial/security/oauth2-jwt.md", "Handle JWT tokens")],
     "Hash passwords with a password-hashing library and issue signed JWT tokens with an expiration."),
    ("q08", P, "run some setup and cleanup logic when my app starts and stops",
     [(F + "advanced/events.md", "Lifespan"), (F + "advanced/events.md", "Lifespan function"), (S + "lifespan.md", "(introduction)")],
     "Use a `lifespan` async context manager: code before `yield` runs at startup, code after runs at shutdown."),
    ("q09", P, "serve my app behind a reverse proxy that strips a path prefix",
     [(F + "advanced/behind-a-proxy.md", "Proxy with a stripped path prefix"), (F + "advanced/behind-a-proxy.md", "Providing the `root_path`")],
     "Set `root_path` (via `--root-path` or the `FastAPI` parameter) so docs and URLs include the stripped prefix."),
    ("q10", P, "send live messages to the browser over a persistent two-way connection",
     [(F + "advanced/websockets.md", "Create a `websocket`"), (F + "advanced/websockets.md", "Await for messages and send messages"), (S + "websockets.md", "WebSocket")],
     "Declare a `@app.websocket` endpoint, `accept()` the connection, then `receive_text()` and `send_text()` in a loop."),
    ("q11", K, "response model",
     [(F + "tutorial/response-model.md", "`response_model` Parameter")],
     "The `response_model` parameter declares the model used to validate, filter and document the response."),
    ("q12", K, "dependency injection",
     [(F + "tutorial/dependencies/index.md", "What is \"Dependency Injection\""), (F + "tutorial/dependencies/index.md", "First Steps"), (F + "features.md", "Dependency Injection")],
     "Dependency injection lets a path operation declare what it needs with `Depends`, and FastAPI provides it."),
    ("q13", K, "field validator",
     [(PY + "concepts/validators.md", "Field validators")],
     "Use the `@field_validator` decorator on a model classmethod to validate or transform a single field."),
    ("q14", K, "custom exception handler",
     [(F + "tutorial/handling-errors.md", "Install custom exception handlers")],
     "Define your exception class and register a handler with `@app.exception_handler(MyException)`."),
    ("q15", P, "add code that runs on every request before and after the endpoint",
     [(F + "tutorial/middleware.md", "Create a middleware"), (F + "tutorial/middleware.md", "Before and after the `response`"), (S + "middleware.md", "Using middleware")],
     "Create a middleware with `@app.middleware(\"http\")` that runs code before and after calling `call_next`."),
    ("q16", P, "give a model attribute a fallback value when it is missing",
     [(PY + "concepts/fields.md", "Default values")],
     "Assign a default value to the field, or use `Field(default=...)` / `default_factory`."),
    ("q17", P, "convert a model instance to a dictionary or a JSON string",
     [(PY + "concepts/serialization.md", "Serializing data"), (PY + "concepts/serialization.md", "Python mode"), (PY + "concepts/serialization.md", "JSON mode")],
     "Use `model_dump()` for a dict and `model_dump_json()` for a JSON string."),
    ("q18", P, "turn off automatic type coercion so that a string is not accepted as an integer",
     [(PY + "concepts/strict_mode.md", "(introduction)"), (PY + "concepts/strict_mode.md", "As a validation parameter"), (PY + "why.md", "Strict mode and data coercion")],
     "Enable strict mode, per validation call, per field or in the model config, so that no type coercion happens."),
    ("q19", P, "pick the right model in a union based on the value of one field",
     [(PY + "concepts/unions.md", "Discriminated unions")],
     "Use a discriminated union with `Field(discriminator='field_name')` so validation selects the member by that field."),
    ("q20", P, "check the types of a plain function's arguments at call time",
     [(PY + "concepts/validation_decorator.md", "(introduction)")],
     "Decorate the function with `@validate_call` so its arguments are validated."),
    ("q21", K, "generate JSON schema from a model",
     [(PY + "concepts/json_schema.md", "Generating JSON Schema")],
     "Call `model_json_schema()` on the model (or `TypeAdapter.json_schema()`)."),
    ("q22", P, "accept a different name for a field in the incoming data",
     [(PY + "concepts/alias.md", "(introduction)"), (PY + "concepts/fields.md", "Field aliases")],
     "Give the field an alias (`Field(alias=...)`, `validation_alias`) so the input key differs from the attribute name."),
    ("q23", P, "test my API endpoints without starting a server",
     [(F + "tutorial/testing.md", "Using `TestClient`"), (S + "testclient.md", "(introduction)")],
     "Use `TestClient(app)` and call it like a requests/httpx client inside pytest tests."),
    ("q24", P, "package my application in a container image",
     [(F + "deployment/docker.md", "Build a Docker Image for FastAPI"), (F + "deployment/docker.md", "Dockerfile")],
     "Write a Dockerfile that installs requirements and starts the app with `fastapi run`, then build the image."),
    ("q25", P, "handle more traffic by running several processes of the server",
     [(F + "deployment/server-workers.md", "Multiple Workers"), (F + "deployment/concepts.md", "Multiple Processes - Workers")],
     "Start the server with multiple worker processes (for example `fastapi run --workers 4`)."),
    ("q26", P, "protect an endpoint so only requests with a valid bearer token can use it",
     [(F + "tutorial/security/first-steps.md", "**FastAPI**'s `OAuth2PasswordBearer`"), (F + "tutorial/security/first-steps.md", "Use it")],
     "Create an `OAuth2PasswordBearer` instance and use it as a dependency to extract the token."),
    ("q27", P, "when should an endpoint be async def and when plain def",
     [(F + "async.md", "In a hurry?"), (F + "async.md", "Path operation functions")],
     "Use `async def` if you call awaitable libraries; use plain `def` for blocking code, which runs in a threadpool."),
    ("q28", P, "return rendered HTML pages from templates",
     [(F + "advanced/templates.md", "Using `Jinja2Templates`"), (S + "templates.md", "(introduction)")],
     "Use `Jinja2Templates` and return `templates.TemplateResponse` with a request and context."),
    ("q29", P, "restrict a numeric path parameter to a range of values",
     [(F + "tutorial/path-params-numeric-validations.md", "Number validations: greater than or eq"), (F + "tutorial/path-params-numeric-validations.md", "Number validations: greater than and l")],
     "Use `Path(ge=..., le=...)` (or `gt`, `lt`) to constrain the numeric value."),
    ("q30", P, "update only some fields of an existing item without replacing the whole thing",
     [(F + "tutorial/body-updates.md", "Partial updates with `PATCH`"), (F + "tutorial/body-updates.md", "Using Pydantic's `exclude_unset` param")],
     "Use PATCH and `model_dump(exclude_unset=True)` to update only the fields the client sent."),
    ("q31", K, "list of submodels nested models",
     [(F + "tutorial/body-nested-models.md", "Nested Models"), (F + "tutorial/body-nested-models.md", "Attributes with lists of submodels")],
     "Declare a model as the type of an attribute, or a `list[SubModel]`, to nest models in the body."),
    ("q32", P, "read configuration values from a .env file",
     [(F + "advanced/settings.md", "Reading a `.env` file"), (F + "advanced/settings.md", "Read settings from `.env`"), (S + "config.md", "(introduction)")],
     "Use a `Settings` class (pydantic-settings) with `env_file` configured, or Starlette's `Config`."),
    ("q33", P, "send a very large response piece by piece instead of loading it in memory",
     [(F + "advanced/stream-data.md", "A `StreamingResponse` with `yield`"), (S + "responses.md", "StreamingResponse")],
     "Return a `StreamingResponse` fed by a generator that yields chunks."),
    ("q34", P, "upgrade my project from pydantic version 1 to version 2",
     [(F + "how-to/migrate-from-pydantic-v1-to-pydantic-v2.md", "(introduction)"), (PY + "migration.md", "Install Pydantic V2"), (PY + "migration.md", "Code transformation tool")],
     "Follow the migration guide and use the `bump-pydantic` code transformation tool."),
    ("q35", K, "swagger ui theme parameters",
     [(F + "how-to/configure-swagger-ui.md", "Change the Theme"), (F + "how-to/configure-swagger-ui.md", "Change Default Swagger UI Parameters")],
     "Pass `swagger_ui_parameters` to the FastAPI app to change Swagger UI settings, including the syntax-highlight theme."),
    # requêtes courtes, style mots-clés / noms d'API (équilibrent le set : sans elles, il favoriserait la recherche dense)
    ("q36", K, "CORS middleware",
     [(F + "tutorial/cors.md", "Use `CORSMiddleware`"), (S + "middleware.md", "CORSMiddleware")],
     "Add `CORSMiddleware` with the allowed origins, methods and headers."),
    ("q37", K, "path parameters enum",
     [(F + "tutorial/path-params.md", "Predefined values"), (F + "tutorial/path-params.md", "Create an `Enum` class")],
     "Create a `str, Enum` class and use it as the type of the path parameter to restrict it to predefined values."),
    ("q38", K, "OAuth2 scopes",
     [(F + "advanced/security/oauth2-scopes.md", "OAuth2 scopes and OpenAPI")],
     "OAuth2 scopes are permission strings declared in the security scheme and checked with `SecurityScopes`."),
    ("q39", K, "HTTPException",
     [(F + "tutorial/handling-errors.md", "Raise an `HTTPException` in your code"), (S + "exceptions.md", "HTTPException")],
     "`HTTPException` carries a status code and detail and is turned into an error response."),
    ("q40", K, "dependencies with yield",
     [(F + "tutorial/dependencies/dependencies-with-yield.md", "A dependency with `yield` and `try`"), (F + "tutorial/dependencies/dependencies-with-yield.md", "A database dependency with `yield`")],
     "A dependency can `yield` a value; code after the `yield` runs as cleanup after the response."),
    ("q41", K, "computed_field",
     [(PY + "concepts/fields.md", "The `computed_field` decorator")],
     "`@computed_field` includes a property in serialization and JSON schema."),
    ("q42", K, "model_validator",
     [(PY + "concepts/validators.md", "Model validators")],
     "`@model_validator` validates the model as a whole, in `before`, `after` or `wrap` mode."),
    ("q43", K, "TypeAdapter",
     [(PY + "concepts/type_adapter.md", "Parsing data into a specified type")],
     "`TypeAdapter` validates and serializes arbitrary types without defining a model."),
    ("q44", K, "GZipMiddleware",
     [(S + "middleware.md", "GZipMiddleware"), (F + "advanced/middleware.md", "`GZipMiddleware`")],
     "`GZipMiddleware` compresses responses for clients that send `Accept-Encoding: gzip`."),
]


def main():
    docs = [json.loads(l) for l in (ROOT / "data" / "corpus.jsonl").open(encoding="utf-8")]
    by_page = {}
    for d in docs:
        by_page.setdefault(d["path"], []).append(d["section"])
    out, errors = [], []
    for qid, kind, query, rel, answer in RAW:
        keys = []
        for page, prefix in rel:
            hits = [s for s in by_page.get(page, []) if s == prefix or (prefix != "(introduction)" and s.startswith(prefix))]
            exact = [s for s in hits if s == prefix]
            hits = exact or hits
            if len(hits) != 1:
                errors.append(f"{qid}: {page} / {prefix!r} -> {hits or 'AUCUNE'} (sections: {by_page.get(page)})")
            else:
                keys.append(f"{page}#{hits[0]}")
        out.append({"id": qid, "kind": kind, "query": query, "relevant": keys, "expected_answer": answer})
    if errors:
        print("\n".join(errors)); sys.exit(1)
    dest = ROOT / "data" / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "queries.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    kinds = {k: sum(q["kind"] == k for q in out) for k in (K, P)}
    print(f"{len(out)} requêtes -> data/eval/queries.json {kinds}, {sum(len(q['relevant']) for q in out)} annotations")


if __name__ == "__main__":
    main()
