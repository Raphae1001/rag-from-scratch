import pytest

from rag.faithfulness import _parse_judge_response, judge_faithfulness
from rag.generate import FakeLLMClient


def test_parse_judge_response_plain_json():
    raw = '{"claims": [{"claim": "x uses y", "supported": true}]}'
    assert _parse_judge_response(raw)["claims"] == [{"claim": "x uses y", "supported": True}]


def test_parse_judge_response_ignores_trailing_prose():
    raw = '{"claims": []}\n\nThe answer had no factual content to check.'
    assert _parse_judge_response(raw)["claims"] == []


def test_parse_judge_response_rejects_non_json():
    with pytest.raises(ValueError, match="non-JSON"):
        _parse_judge_response("I cannot evaluate this.")


def test_parse_judge_response_rejects_missing_claims_key():
    with pytest.raises(ValueError, match="claims"):
        _parse_judge_response('{"result": "ok"}')


def test_parse_judge_response_rejects_claim_without_boolean_supported():
    with pytest.raises(ValueError, match="supported"):
        _parse_judge_response('{"claims": [{"claim": "x", "supported": "yes"}]}')


def test_judge_faithfulness_all_claims_supported():
    client = FakeLLMClient({
        "declare a BackgroundTasks": '{"claims": ['
            '{"claim": "Declare a BackgroundTasks parameter", "supported": true}, '
            '{"claim": "Call add_task to schedule work", "supported": true}]}',
    })
    result = judge_faithfulness(client, "declare a BackgroundTasks parameter and call add_task",
                                 [("fastapi/tutorial/background-tasks.md#intro", "some passage text")])
    assert result == {"score": 1.0, "claims": [
        {"claim": "Declare a BackgroundTasks parameter", "supported": True},
        {"claim": "Call add_task to schedule work", "supported": True},
    ]}


def test_judge_faithfulness_partial_support_computes_fraction():
    client = FakeLLMClient({
        "mixed answer": '{"claims": ['
            '{"claim": "a", "supported": true}, '
            '{"claim": "b", "supported": false}, '
            '{"claim": "c", "supported": true}, '
            '{"claim": "d", "supported": false}]}',
    })
    result = judge_faithfulness(client, "mixed answer", [("k", "v")])
    assert result["score"] == 0.5


def test_judge_faithfulness_no_claims_scores_one():
    client = FakeLLMClient({"trivial answer": '{"claims": []}'})
    result = judge_faithfulness(client, "trivial answer", [("k", "v")])
    assert result == {"score": 1.0, "claims": []}
