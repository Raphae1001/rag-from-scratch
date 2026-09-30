"""API FastAPI (Phase 5) : expose le pipeline RAG complet (retrieval hybride -> reranking -> génération
sourcée) via `POST /query`, avec traçage Langfuse par étape si `$LANGFUSE_PUBLIC_KEY` est présent.

Lancer : uvicorn rag.api:app --host 0.0.0.0 --port 8000  (voir aussi le Dockerfile)
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag import db
from rag.embed import SentenceTransformerEmbedder
from rag.generate import AnthropicClient
from rag.index import build_index
from rag.pipeline import answer as run_pipeline
from rag.rerank import CrossEncoderReranker

CORPUS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "corpus.jsonl"
STATE: dict = {}


def _init_langfuse():
    """None si $LANGFUSE_PUBLIC_KEY absent : le traçage est une fonctionnalité optionnelle, pas un
    prérequis pour que l'API réponde (cohérent avec le reste du projet : pgserver optionnel, etc.)."""
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    from langfuse import Langfuse  # import tardif : dépendance optionnelle

    return Langfuse()  # lit LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST dans l'environnement


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge tout une seule fois au démarrage (modèles, index, connexion DB) plutôt qu'à chaque requête —
    le chargement des modèles domine largement la latence d'une requête isolée (mesuré en Phase 3)."""
    docs = [json.loads(line) for line in CORPUS_PATH.open(encoding="utf-8")]
    STATE["index"] = build_index([d["text"] for d in docs])
    STATE["conn"] = db.connect(os.environ.get("DATABASE_URL"))
    n_docs, _ = db.counts(STATE["conn"])
    if n_docs != len(docs):
        raise RuntimeError(f"embeddings non chargés : {n_docs} docs en base, {len(docs)} dans le corpus "
                            "— lancer scripts/embed_corpus.py")
    STATE["embedder"] = SentenceTransformerEmbedder()
    STATE["reranker"] = CrossEncoderReranker()
    STATE["client"] = AnthropicClient()
    STATE["langfuse"] = _init_langfuse()
    yield
    STATE["conn"].close()
    if STATE["langfuse"]:
        STATE["langfuse"].flush()


app = FastAPI(title="RAG from First Principles", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    top_n: int = 20


class QueryResponse(BaseModel):
    answerable: bool
    answer: str | None
    citations: list[str]
    latency_ms: dict[str, float]
    trace_url: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


def _traced_query(req: QueryRequest) -> tuple[dict, str | None]:
    """Exécute le pipeline et l'enregistre dans Langfuse : une observation par étape (retrieval, reranking,
    génération), chacune avec sa vraie latence mesurée en métadonnée — pas la durée de l'observation
    Langfuse elle-même (créée après coup, une fois le pipeline terminé), qui serait quasi nulle."""
    langfuse = STATE["langfuse"]
    with langfuse.start_as_current_observation(name="query", as_type="span", input={"query": req.query}) as root:
        result = run_pipeline(req.query, STATE["conn"], STATE["index"], STATE["embedder"],
                               STATE["reranker"], STATE["client"], k=req.k, top_n=req.top_n)
        lat = result["latency_ms"]

        with langfuse.start_as_current_observation(
            name="retrieval", as_type="retriever", input={"query": req.query},
            metadata={"latency_ms": lat["retrieval"], "top_n": req.top_n},
        ):
            pass

        with langfuse.start_as_current_observation(
            name="reranking", as_type="span", metadata={"latency_ms": lat["reranking"], "k": req.k},
        ) as sp:
            sp.update(output={"citations": result["citations"]})

        with langfuse.start_as_current_observation(
            name="generation", as_type="generation", model=STATE["client"].model,
            input={"query": req.query}, metadata={"latency_ms": lat["generation"]},
            usage_details=STATE["client"].last_usage,
        ) as sp:
            sp.update(output=result["answer"])

        root.update(output={"answerable": result["answerable"], "citations": result["citations"]},
                    metadata={"latency_ms": lat})
        trace_id = langfuse.get_current_trace_id()
    langfuse.flush()
    return result, langfuse.get_trace_url(trace_id=trace_id)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query ne peut pas être vide")

    if STATE["langfuse"] is None:
        result = run_pipeline(req.query, STATE["conn"], STATE["index"], STATE["embedder"],
                               STATE["reranker"], STATE["client"], k=req.k, top_n=req.top_n)
        trace_url = None
    else:
        result, trace_url = _traced_query(req)

    return QueryResponse(answerable=result["answerable"], answer=result["answer"],
                          citations=result["citations"], latency_ms=result["latency_ms"], trace_url=trace_url)
