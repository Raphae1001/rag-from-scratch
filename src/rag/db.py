"""Accès Postgres + pgvector : chargement des embeddings et recherche dense."""
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

DEFAULT_DSN = "postgresql://rag:rag@localhost:5433/rag"
INIT_SQL = Path(__file__).resolve().parent.parent.parent / "db" / "init.sql"
HNSW_INDEX = "chunks_embedding_hnsw"


def connect(dsn: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(dsn or os.environ.get("DATABASE_URL", DEFAULT_DSN), autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(INIT_SQL.read_text(encoding="utf-8"))


def reset_data(conn: psycopg.Connection) -> None:
    conn.execute("TRUNCATE chunks, docs RESTART IDENTITY CASCADE")


def load_corpus(conn, docs: list[dict], chunks: list[tuple[int, int, str, np.ndarray]]) -> None:
    """docs : dicts du corpus (avec 'id') ; chunks : (doc_id, chunk_no, texte, embedding)."""
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO docs (doc_id, source, path, page, section, text) VALUES (%s,%s,%s,%s,%s,%s)",
            [(d["id"], d["source"], d["path"], d["page"], d["section"], d["text"]) for d in docs],
        )
        cur.executemany(
            "INSERT INTO chunks (doc_id, chunk_no, text, embedding) VALUES (%s,%s,%s,%s)",
            [(doc_id, no, text, emb) for doc_id, no, text, emb in chunks],
        )


def counts(conn) -> tuple[int, int]:
    return (conn.execute("SELECT count(*) FROM docs").fetchone()[0],
            conn.execute("SELECT count(*) FROM chunks").fetchone()[0])


def dense_search(conn, qvec: np.ndarray, k: int = 10, candidate_chunks: int | None = None,
                 ef_search: int | None = None) -> list[tuple[int, float]]:
    """Top-k documents par similarité cosinus, le score d'un document étant celui de son meilleur passage.

    On récupère les `candidate_chunks` passages les plus proches (4k par défaut, min. 40) puis on garde
    le meilleur par document. Si l'index HNSW existe, `hnsw.ef_search` doit être >= au nombre de
    passages demandés, sinon Postgres en renverrait moins : on le règle en conséquence.
    """
    n = candidate_chunks or max(4 * k, 40)
    conn.execute(f"SET hnsw.ef_search = {int(max(n, ef_search or 0))}")
    rows = conn.execute(
        "SELECT doc_id, embedding <=> %s AS dist FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (qvec, qvec, n),
    ).fetchall()
    best: dict[int, float] = {}
    for doc_id, dist in rows:            # rows est trié par distance croissante : le 1er vu est le meilleur
        best.setdefault(doc_id, dist)
    return [(d, 1.0 - dist) for d, dist in list(best.items())[:k]]


def create_hnsw(conn, m: int = 16, ef_construction: int = 64) -> None:
    conn.execute(f"CREATE INDEX IF NOT EXISTS {HNSW_INDEX} ON chunks USING hnsw (embedding vector_cosine_ops) "
                 f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)})")


def drop_hnsw(conn) -> None:
    conn.execute(f"DROP INDEX IF EXISTS {HNSW_INDEX}")


def has_hnsw(conn) -> bool:
    return conn.execute("SELECT 1 FROM pg_indexes WHERE indexname = %s", (HNSW_INDEX,)).fetchone() is not None


@contextmanager
def exact_search(conn):
    """Force le parcours séquentiel (recherche EXACTE) même si l'index HNSW existe, puis restaure les réglages.

    Désactiver seulement enable_indexscan ne suffit pas si enable_seqscan est lui aussi coupé :
    Postgres retombe alors sur l'index (coût pénalisé mais choisi). On active donc explicitement le seqscan.
    """
    saved = {name: conn.execute(f"SHOW {name}").fetchone()[0] for name in ("enable_seqscan", "enable_indexscan")}
    conn.execute("SET enable_seqscan = on")
    conn.execute("SET enable_indexscan = off")
    try:
        yield
    finally:
        for name, value in saved.items():
            conn.execute(f"SET {name} = {value}")
