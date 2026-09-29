CREATE EXTENSION IF NOT EXISTS vector;

-- une ligne par document du corpus (doc_id = position dans data/corpus.jsonl)
CREATE TABLE IF NOT EXISTS docs (
    doc_id  integer PRIMARY KEY,
    source  text NOT NULL,
    path    text NOT NULL,
    page    text,
    section text NOT NULL,
    text    text NOT NULL
);

-- une ligne par passage embeddé (les documents longs ont plusieurs passages, voir src/rag/chunking.py)
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  serial PRIMARY KEY,
    doc_id    integer NOT NULL REFERENCES docs(doc_id) ON DELETE CASCADE,
    chunk_no  integer NOT NULL,
    text      text NOT NULL,
    embedding vector(384) NOT NULL,          -- all-MiniLM-L6-v2, vecteurs normalisés
    UNIQUE (doc_id, chunk_no)
);
-- L'index HNSW n'est PAS créé ici : il est créé/supprimé par scripts/benchmark.py pour comparer
-- recherche exacte et approximative (voir README, section Phase 2).
