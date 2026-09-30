"""Tests de l'API FastAPI (Phase 5), sur la vraie base de test (`conn`, voir conftest.py) mais avec des
composants factices (HashingEmbedder, OverlapReranker, FakeLLMClient) : pas de modèle téléchargé, pas
d'appel réseau. `rag.api.STATE` est peuplé directement, sans passer par le vrai `lifespan` (qui charge des
modèles réels et exige `$DATABASE_URL`/`$ANTHROPIC_API_KEY`)."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_db import EMB, load, mk  # noqa: E402

from rag import api  # noqa: E402
from rag.generate import FakeLLMClient
from rag.index import build_index
from rag.rerank import OverlapReranker

DOCS = [mk(0, "Errors", "raise an HTTPException with a status code to return an error to the client")]


@pytest.fixture()
def client(conn):
    load(conn, DOCS)
    api.STATE.clear()
    api.STATE.update({
        "index": build_index([d["text"] for d in DOCS]),
        "conn": conn,
        "embedder": EMB,
        "reranker": OverlapReranker(),
        "client": FakeLLMClient({
            "HTTPException": '{"answerable": true, "answer": "Raise HTTPException.", "sources": ["S1"]}',
        }),
        "langfuse": None,
    })
    return TestClient(api.app)


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_query_returns_answer_with_citations_and_latency(client):
    res = client.post("/query", json={"query": "raise an HTTPException error"})
    assert res.status_code == 200
    body = res.json()
    assert body["answerable"] is True
    assert body["answer"] == "Raise HTTPException."
    assert body["citations"] == ["t/p0.md#Errors"]
    assert set(body["latency_ms"]) == {"retrieval", "reranking", "generation"}
    assert body["trace_url"] is None  # pas de Langfuse configuré dans ce test


def test_query_refusal_has_no_citations(client):
    api.STATE["client"] = FakeLLMClient({"capital of France": '{"answerable": false}'})
    res = client.post("/query", json={"query": "what is the capital of France"})
    body = res.json()
    assert body["answerable"] is False
    assert body["answer"] is None
    assert body["citations"] == []


def test_query_rejects_empty_query(client):
    res = client.post("/query", json={"query": "   "})
    assert res.status_code == 422


def test_query_rejects_missing_query_field(client):
    res = client.post("/query", json={})
    assert res.status_code == 422
