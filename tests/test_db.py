"""Tests sur un VRAI Postgres + pgvector. Fixtures DB (`dsn`, `conn`) dans conftest.py, partagées avec
test_api.py. Les embeddings sont ceux du HashingEmbedder (lexicaux) : on teste la plomberie SQL, pas la
sémantique."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("psycopg")
pytest.importorskip("pgvector")

from rag import db  # noqa: E402
from rag.chunking import make_chunks  # noqa: E402
from rag.embed import HashingEmbedder  # noqa: E402
from rag.generate import FakeLLMClient  # noqa: E402
from rag.index import build_index  # noqa: E402
from rag.pipeline import answer  # noqa: E402
from rag.rerank import OverlapReranker  # noqa: E402
from rag.retrieval import search_bm25, search_dense, search_hybrid  # noqa: E402

EMB = HashingEmbedder()
CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus.jsonl"


def load(conn, docs):
    chunks = []
    for d in docs:
        pieces = make_chunks(d, size=30, overlap=5)
        vecs = EMB.encode(pieces)
        chunks += [(d["id"], i, p, v) for i, (p, v) in enumerate(zip(pieces, vecs))]
    db.load_corpus(conn, docs, chunks)
    return chunks


def mk(i, section, text):
    return {"id": i, "source": "t", "path": f"t/p{i}.md", "page": f"Page {i}", "section": section,
            "text": f"Page {i}\n{section}\n{text}"}


DOCS = [
    mk(0, "Errors", "raise an HTTPException with a status code to return an error to the client"),
    mk(1, "Middleware", "a middleware runs code before and after every request handled by the application"),
    mk(2, "Files", "upload a file with UploadFile and File parameters in the path operation"),
    # document long : le mot rare n'apparaît qu'à la toute fin (au-delà de la fenêtre du 1er passage)
    mk(3, "Long page", " ".join(["filler"] * 200) + " zebrafish migration pattern"),
]


def test_schema_and_counts(conn):
    chunks = load(conn, DOCS)
    n_docs, n_chunks = db.counts(conn)
    assert n_docs == 4
    assert n_chunks == len(chunks) > 4            # le document long est découpé en plusieurs passages


def test_dense_search_finds_the_matching_document(conn):
    load(conn, DOCS)
    res = db.dense_search(conn, EMB.encode(["return an error status code"])[0], k=3)
    assert res[0][0] == 0
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 + 1e-6 for s in scores)


def test_one_result_per_document_with_best_chunk_score(conn):
    load(conn, DOCS)
    res = db.dense_search(conn, EMB.encode(["filler"])[0], k=10)
    ids = [d for d, _ in res]
    assert len(ids) == len(set(ids))              # le doc 3 a beaucoup de passages mais n'apparaît qu'une fois


def test_chunking_makes_the_tail_of_long_documents_searchable(conn):
    load(conn, DOCS)
    res = db.dense_search(conn, EMB.encode(["zebrafish migration pattern"])[0], k=3)
    assert res[0][0] == 3


def test_hnsw_index_matches_exact_search_on_small_data(conn):
    load(conn, DOCS)
    q = EMB.encode(["upload a file"])[0]
    exact = db.dense_search(conn, q, k=4)
    db.create_hnsw(conn)
    assert db.has_hnsw(conn)
    conn.execute("SET enable_seqscan = off")       # forcer le planificateur à utiliser l'index
    plan = " ".join(r[0] for r in conn.execute(
        "EXPLAIN SELECT doc_id FROM chunks ORDER BY embedding <=> %s LIMIT 5", (q,)).fetchall())
    assert "hnsw" in plan
    approx = db.dense_search(conn, q, k=4)
    assert approx[0][0] == exact[0][0] == 2
    # mêmes scores (l'ordre des ex æquo à similarité ~0 est arbitraire, donc on compare les scores)
    assert [round(s, 5) for _, s in approx] == [round(s, 5) for _, s in exact]
    assert {d for d, _ in approx} == {d for d, _ in exact}
    conn.execute("SET enable_seqscan = on")


def test_exact_search_mode_bypasses_the_index(conn):
    load(conn, DOCS)
    db.create_hnsw(conn)
    q = EMB.encode(["middleware"])[0]
    conn.execute("SET enable_seqscan = off")
    plan_sql = "EXPLAIN SELECT doc_id FROM chunks ORDER BY embedding <=> %s LIMIT 5"
    before = " ".join(r[0] for r in conn.execute(plan_sql, (q,)).fetchall())
    assert "hnsw" in before                        # sans le mode exact, l'index est bien utilisé
    with db.exact_search(conn):
        inside = " ".join(r[0] for r in conn.execute(plan_sql, (q,)).fetchall())
        assert "hnsw" not in inside and "Seq Scan" in inside
    after = " ".join(r[0] for r in conn.execute(plan_sql, (q,)).fetchall())
    assert "hnsw" in after                         # le mode exact est bien restauré à la sortie
    conn.execute("SET enable_seqscan = on")


def test_get_docs_returns_path_section_text_by_doc_id(conn):
    load(conn, DOCS)
    docs = db.get_docs(conn, [2, 0])
    assert set(docs) == {0, 2}
    assert docs[2]["path"] == "t/p2.md"
    assert docs[2]["section"] == "Files"
    assert "UploadFile" in docs[2]["text"]
    assert docs[0]["section"] == "Errors"


def test_get_docs_empty_list_returns_empty_dict(conn):
    load(conn, DOCS)
    assert db.get_docs(conn, []) == {}


def test_hybrid_search_combines_both_rankings(conn):
    load(conn, DOCS)
    index = build_index([d["text"] for d in DOCS])
    res = search_hybrid(index, conn, EMB, "raise an HTTPException error", k=4)
    assert res[0] == 0
    assert set(res) <= {0, 1, 2, 3}
    assert search_bm25(index, "HTTPException", k=3)[0] == 0
    assert search_dense(conn, EMB, "HTTPException", k=3)[0] == 0


def test_pipeline_answer_returns_citations_and_latency(conn):
    """rag.pipeline.answer : le pipeline complet (retrieval -> reranking -> génération) partagé par
    scripts/answer.py et l'API (Phase 5). FakeLLMClient + OverlapReranker : pas d'appel réseau."""
    load(conn, DOCS)
    index = build_index([d["text"] for d in DOCS])
    client = FakeLLMClient({
        "raise an HTTPException": '{"answerable": true, "answer": "Raise HTTPException.", "sources": ["S1"]}',
    })
    result = answer("raise an HTTPException error", conn, index, EMB, OverlapReranker(), client, k=3, top_n=4)

    assert result["answerable"] is True
    assert result["citations"] == ["t/p0.md#Errors"]
    assert set(result["latency_ms"]) == {"retrieval", "reranking", "generation"}
    assert all(ms >= 0 for ms in result["latency_ms"].values())


@pytest.mark.skipif(not CORPUS.exists(), reason="lancer scripts/build_corpus.py d'abord")
def test_whole_corpus_loads_and_every_document_has_a_chunk(conn):
    docs = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
    load(conn, docs)
    n_docs, n_chunks = db.counts(conn)
    assert n_docs == len(docs) >= 1000
    assert n_chunks >= n_docs
    orphans = conn.execute("SELECT count(*) FROM docs d LEFT JOIN chunks c USING (doc_id) WHERE c.doc_id IS NULL").fetchone()[0]
    assert orphans == 0
    dim = conn.execute("SELECT vector_dims(embedding) FROM chunks LIMIT 1").fetchone()[0]
    assert dim == 384
