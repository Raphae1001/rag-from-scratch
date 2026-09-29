"""Le set de non-réponse (Phase 3) doit rester structurellement valide (spec : au moins 10 requêtes)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NO_ANSWER = ROOT / "data" / "eval" / "no_answer.json"
pytestmark = pytest.mark.skipif(not NO_ANSWER.exists(), reason="set de non-réponse absent")


@pytest.fixture(scope="module")
def queries():
    return json.loads(NO_ANSWER.read_text(encoding="utf-8"))


def test_at_least_ten_queries(queries):
    assert len(queries) >= 10                                 # exigence de la spec (Phase 3)


def test_ids_unique_and_fields_present(queries):
    assert len({q["id"] for q in queries}) == len(queries)
    for q in queries:
        assert q["query"].strip()


def test_no_overlap_with_the_annotated_eval_set(queries):
    """Ces requêtes ne doivent pas être un doublon du set annoté de la Phase 4 (sinon le test de
    non-réponse mesurerait la même chose que le recall, pas la détection de hors-contexte)."""
    eval_path = ROOT / "data" / "eval" / "queries.json"
    if not eval_path.exists():
        pytest.skip("set annoté absent")
    answerable = {q["query"] for q in json.loads(eval_path.read_text(encoding="utf-8"))}
    overlap = {q["query"] for q in queries} & answerable
    assert not overlap, overlap
