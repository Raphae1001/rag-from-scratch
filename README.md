# RAG From Scratch

An end-to-end Retrieval-Augmented Generation system built from first principles: a BM25 search engine
implemented from scratch, dense retrieval with pgvector, hybrid fusion, sourced generation with an
anti-hallucination guardrail, LLM-judged evaluation, a production API with tracing, and contrastive
fine-tuning of the embedding model.

**Why this exists**: this is a learning and portfolio project, built in six phases, each with its own
measured "gate" (a numeric bar the phase has to clear before moving on) rather than a vibe check. The
goal was to understand what's actually happening inside a RAG pipeline — not to import a framework and
call `retriever.query()`. Every core piece (the inverted index, BM25 scoring, RRF fusion, the
retrieval → rerank → generate pipeline) is written directly against numpy/Postgres/the Claude API, with
no LangChain/LlamaIndex-style abstraction layer in between.

**Corpus**: the official documentation of [FastAPI](https://github.com/fastapi/fastapi),
[Starlette](https://github.com/encode/starlette) and [Pydantic](https://github.com/pydantic/pydantic) —
the stack this project's author uses day to day — split into 1,616 documents (one per `##`/`###` section),
fetched at pinned commits for a reproducible corpus. Changelogs, "meta" pages (contributing, people,
newsletter) and Pydantic's `api/` stub pages (auto-generated `::: pydantic.X` references with no prose)
are excluded — see `scripts/build_corpus.py`.

## Architecture

```
query ──▶ BM25 (inverted index)   ──┐
      └─▶ dense search (pgvector) ──┴─▶ RRF fusion ──▶ cross-encoder rerank ──▶ Claude (sourced answer)
```

1. **Lexical retrieval** — a BM25 inverted index built from scratch (`src/rag/index.py`, `src/rag/bm25.py`).
2. **Dense retrieval** — passages embedded with `all-MiniLM-L6-v2` (`src/rag/embed.py`), stored and
   searched by cosine similarity in Postgres/pgvector (`src/rag/db.py`), with an optional HNSW index.
3. **Fusion** — Reciprocal Rank Fusion merges the two rankings by rank only, so no score normalization is
   needed between BM25's and cosine's different scales (`src/rag/fusion.py`).
4. **Reranking** — a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the fused
   candidates against the actual query text (`src/rag/rerank.py`).
5. **Generation** — the top passages are labeled `[S1]`, `[S2]`, ... and handed to Claude, which must
   answer in strict JSON: either `{"answerable": false}` or `{"answerable": true, "answer": ..., "sources": [...]}`
   (`src/rag/generate.py`). This is the anti-hallucination guardrail: a structured refusal is
   deterministic to detect, unlike hoping a free-text answer starts with "I don't know".
6. **Serving** — a FastAPI service (`src/rag/api.py`) wires all of the above behind `POST /query`, with
   optional Langfuse tracing per request.
7. **Fine-tuning** — a LoRA adapter trained with a contrastive loss on (section title → section body)
   pairs improves the dense embeddings without touching the base model's weights.

Everything that talks to a model or the network (`Embedder`, `Reranker`, `LLMClient`) is a `Protocol`
with a deterministic fake implementation (`HashingEmbedder`, `OverlapReranker`, `FakeLLMClient`) used in
tests and in a `--fake` pipeline-smoke-test mode, so the retrieval/fusion/parsing logic can be tested
without downloading a model or spending API credits.

## Quickstart

```bash
bash scripts/fetch_docs.sh          # clone the 3 doc repos at pinned commits (reproducible corpus)
python scripts/build_corpus.py      # -> data/corpus.jsonl (1,616 documents)
pip install -r requirements-dev.txt
pytest                              # ~147 tests; DB tests use Docker if `pgserver` isn't installed
python scripts/phase1_check.py      # rebuilds the BM25 index + runs 5 sanity queries
```

Pinned commits: FastAPI `a3d205b`, Starlette `63c5760`, Pydantic `bb6da4c` (see `scripts/fetch_docs.sh`).

That gets you BM25-only search. To run the full hybrid/generation/API pipeline:

```bash
docker compose up -d --wait          # Postgres 16 + pgvector on localhost:5433
python scripts/embed_corpus.py       # ~2,750 passages -> `chunks` table (a few minutes on CPU)
export ANTHROPIC_API_KEY=...         # generation needs a Claude API key
python scripts/answer.py "how do I upload a file to the server"
```

or the API:

```bash
docker compose up -d --wait          # starts Postgres AND the API; loads embeddings on first boot if missing
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "how do I upload a file to the server"}'
```

Without Docker, both `embed_corpus.py` and `benchmark.py` accept `--pgdata data/pgdata` to run a local
Postgres+pgvector via the `pgserver` pip package instead — but `pgserver` ships no wheel for Python ≥ 3.13,
so on 3.13+ Docker is the only option (handled gracefully: `pip install -r requirements-dev.txt` doesn't
fail outright, it just silently skips that one optional dependency).

## Project layout

```
src/rag/         core library (import as `rag.*`, PYTHONPATH=src)
  tokenizer.py     Unicode word tokenizer for BM25
  index.py         inverted index (term -> {doc_id: freq})
  bm25.py          IDF + BM25 scoring + top-k search
  chunking.py      word-window passage splitting for embeddings
  embed.py         Embedder protocol: SentenceTransformerEmbedder (real) / HashingEmbedder (fake)
  db.py            Postgres/pgvector access: load, dense search, HNSW index management
  fusion.py        Reciprocal Rank Fusion
  retrieval.py     the 3 compared configs: BM25-only, dense-only, hybrid
  rerank.py        Reranker protocol: CrossEncoderReranker (real) / OverlapReranker (fake)
  generate.py      LLMClient protocol, JSON-guarded sourced generation
  faithfulness.py  LLM-judged faithfulness scoring (claim-by-claim)
  metrics.py       recall@k, precision@k, MRR
  pipeline.py      retrieval -> rerank -> generate, with per-stage latency
  api.py           FastAPI app (POST /query, GET /health) + Langfuse tracing

scripts/         CLIs and one-off tools (run with `python scripts/<name>.py`)
  fetch_docs.sh, build_corpus.py      build the corpus from the pinned doc repos
  embed_corpus.py, ensure_embedded.py chunk + embed + load into Postgres (the latter is idempotent,
                                       used by the Docker entrypoint)
  answer.py                           CLI: ask the full pipeline one question
  benchmark.py, bootstrap_ci.py       Phase 2 retrieval benchmark + bootstrap confidence interval
  phase1_check.py, phase3_check.py,
  phase4_check.py, phase6_check.py    per-phase gate verification -> results/phaseN_report.{md,json}
  make_eval_set.py                    source of truth for data/eval/queries.json (44 annotated queries)
  make_training_pairs.py,
  finetune_embeddings.py              Phase 6: training pairs + LoRA contrastive fine-tuning
  measure_latency.py                  warm + repeated latency measurement against a running API
  run_phase2.sh                       one-shot: venv, tests, Docker, embeddings, benchmark
  _common.py                          shared paths / corpus loading / DB-arg parsing helpers

data/            corpus.jsonl, eval/ (annotated queries + no-answer set), train/ (fine-tuning pairs)
models/          finetuned-minilm-lora/ — the LoRA checkpoint (1.3 MB, versioned in git)
results/         one {phaseN}_report.{md,json} (or _benchmark/_latency) per phase, all regeneratable
tests/           pytest suite (unit tests + corpus/DB integration tests, skipped gracefully if the
                 corpus or a database isn't available)
db/init.sql      Postgres schema (docs, chunks; the HNSW index is created/dropped by benchmark.py)
```

## Key design decisions

- **Tokenizer**: lowercases, splits on anything that isn't a Unicode letter or digit — including `_`, so
  `response_model` tokenizes to `["response", "model"]` and a query for "response model" finds it.
- **BM25**: the Lucene IDF variant `ln(1 + (N - n_t + 0.5)/(n_t + 0.5))`, which stays non-negative (unlike
  the original Robertson-Spärck Jones formula, which goes negative for very common terms). Verified
  against the `rank_bm25` reference library to <1e-9 with its IDF aligned to ours, and to ≥8/10 ranking
  overlap with its native IDF (`tests/test_reference_rank_bm25.py`).
- **Chunking**: 120-word passages with 20-word overlap, because `all-MiniLM-L6-v2` truncates at 256 word
  pieces and ~25% of corpus sections exceed that window. A document's dense score is its best passage's
  score.
- **Fusion**: RRF (k=60) instead of score blending, specifically because it depends only on rank, not on
  reconciling BM25's ~0–20 scale with cosine's ~0–1 scale.
- **HNSW is not built by default**: `db/init.sql` creates no vector index; `scripts/benchmark.py` builds
  and drops it on demand to compare exact vs. approximate search. At this corpus's scale (~2,750
  passages) exact search is already fast (~4 ms); HNSW becomes worthwhile at larger scale.
- **Anti-hallucination guardrail**: the model answers in strict JSON and never sees the real
  `path#section` citation keys — only `[S1]`, `[S2]`, ... — which are mapped back to real keys after the
  call, so the model can't fabricate a plausible-looking-but-wrong citation.
- **Faithfulness vs. correctness**: the Phase 4 judge checks that every claim in an answer is grounded in
  the *cited* passages — not that the answer is factually right. A second, independent LLM call breaks
  the answer into claims and checks each one (string matching would miss valid paraphrases).
- **Everything network/model-facing is a `Protocol`**: `Embedder`, `Reranker`, `LLMClient` each have a
  real implementation and a deterministic fake one, so the retrieval, fusion, JSON-parsing and citation
  logic are unit-testable with no network calls, no downloaded models, and no flakiness.
- **Langfuse tracing is optional**: the API works with no tracing at all if `$LANGFUSE_PUBLIC_KEY` is
  unset — same pattern as `pgserver` being an optional dev dependency.
- **LoRA, not full fine-tuning**: rank-16 adapters on the attention `query`/`value` projections, base
  model frozen — 0.65% of parameters trainable (147k/22.9M), ~18s to train on CPU, and the result is
  still a standard `sentence-transformers` model, loadable with no extra code.

## Results (measured, not estimated)

Retrieval, 44 annotated queries (16 keyword-style, 28 paraphrased), 1,616 documents:

| Configuration | recall@10 | recall@5 | precision@10 | MRR | latency |
|---|---|---|---|---|---|
| BM25 only | 0.576 | 0.379 | 0.109 | 0.357 | 0.7 ms |
| Dense only | 0.617 | 0.432 | 0.120 | 0.412 | 21.9 ms |
| **Hybrid (RRF)** | **0.739** | **0.485** | **0.141** | **0.508** | 20.0 ms |

The hybrid's +0.163 recall@10 gain over BM25 is statistically significant: a 10,000-resample bootstrap
puts the 95% CI of the gap at **[0.087, 0.250]**, which excludes 0 (`scripts/bootstrap_ci.py`).

Reranking (cross-encoder, 30 candidates) improves MRR (0.508 → 0.556, the first relevant hit ranks
higher) but slightly hurts recall@10 (0.739 → 0.705) — reported as a genuine, documented trade-off rather
than smoothed over.

Generation, on a 12-query out-of-domain set (OAuth2 for Rails, pandas, Kubernetes, "capital of France",
etc.): **0/12 hallucinations**. On the 44 in-domain queries: **0.949** raw LLM-judged faithfulness
(42 answerable, 2 correctly refused); a manual review of every flagged claim across two independent runs
puts the corrected score at **0.957–0.962**, with the flagged gap being under-grounding (a true but not
explicitly-cited fact) rather than outright fabrication in every case reviewed.

Fine-tuning (Phase 6): the LoRA adapter lifts dense-only recall@10 from 0.617 to **0.686** (+0.068), but
the hybrid barely moves (0.739 → 0.735) — RRF was already recovering most of that gap via BM25, so the
fine-tuned dense signal is mostly redundant once fused.

Full per-phase methodology, all the numbers, and the two real bugs found and fixed along the way (a
corpus gap where ~23% of FastAPI documents were missing their code examples; a generation prompt that let
the model ask a clarifying question instead of answering) are in `results/*.md` and the project's commit
history.

## Known limitations

- No stemming or synonym handling in BM25 ("validate" won't match "validation") — this is the gap dense
  retrieval and fusion are meant to cover, and it's the reason a couple of ambiguous queries still fail.
- The 44-query evaluation set is small and self-annotated (by the author, without looking at BM25's
  results first) — the bootstrap CI quantifies sampling noise, not annotation bias.
- No retry/backoff on the Claude API call — a network hiccup fails the whole request.
- No authentication on the API (`/query` is open) — out of scope for a learning project.
- Latency numbers are indicative (Phase 2: single run; Phase 5: 15 requests with warmup) — not a
  production-grade load test.
- One residual unresolved code-include directive in the corpus
  (`fastapi/how-to/configure-swagger-ui.md`, which references FastAPI's own source rather than a
  `docs_src/` example) — everything else was fixed at the source in `build_corpus.py`.
- No horizontal scale, no agentic behavior (single retrieve → rerank → generate turn) — deliberately out
  of scope.

## License and credits

Project code is MIT-licensed (`LICENSE`). The corpus is derived from the official documentation of
FastAPI, Starlette and Pydantic, redistributed under their original licenses (MIT / BSD-3-Clause); see
[`NOTICE.md`](NOTICE.md) and `third_party_licenses/`. This project is not affiliated with or endorsed by
any of them. The corpus is fully regeneratable with `bash scripts/fetch_docs.sh && python scripts/build_corpus.py`.
