"""Le set d'évaluation annoté doit rester cohérent avec le corpus (sinon les métriques sont fausses)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORPUS, EVAL = ROOT / "data" / "corpus.jsonl", ROOT / "data" / "eval" / "queries.json"
pytestmark = pytest.mark.skipif(not (CORPUS.exists() and EVAL.exists()), reason="corpus ou set d'évaluation absent")


@pytest.fixture(scope="module")
def data():
    docs = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
    return {f"{d['path']}#{d['section']}" for d in docs}, json.loads(EVAL.read_text(encoding="utf-8"))


def test_doc_keys_are_unique():
    docs = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
    keys = [f"{d['path']}#{d['section']}" for d in docs]
    assert len(keys) == len(set(keys))


def test_eval_set_size_and_fields(data):
    _, queries = data
    assert 30 <= len(queries) <= 50                       # exigence de la spec (Phase 4)
    assert len({q["id"] for q in queries}) == len(queries)
    for q in queries:
        assert q["query"].strip() and q["expected_answer"].strip()
        assert q["kind"] in {"keyword", "paraphrase"}
        assert len(q["relevant"]) >= 1 and len(set(q["relevant"])) == len(q["relevant"])


def test_every_annotation_exists_in_the_corpus(data):
    keys, queries = data
    missing = [(q["id"], r) for q in queries for r in q["relevant"] if r not in keys]
    assert not missing, missing


def test_set_is_balanced_between_kinds(data):
    _, queries = data
    kw = sum(q["kind"] == "keyword" for q in queries)
    assert 0.25 <= kw / len(queries) <= 0.6               # ni tout mots-clés, ni tout paraphrases
