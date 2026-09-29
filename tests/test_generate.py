import pytest

from rag.generate import FakeLLMClient, _parse_response, build_context, generate_answer


def test_build_context_formats_labeled_passages():
    ctx = build_context([("S1", "hello"), ("S2", "world")])
    assert ctx == "[S1] hello\n\n[S2] world"


def test_parse_response_plain_json():
    assert _parse_response('{"answerable": false}') == {"answerable": False}


def test_parse_response_strips_markdown_fence():
    raw = "```json\n{\"answerable\": true, \"answer\": \"x\", \"sources\": [\"S1\"]}\n```"
    assert _parse_response(raw)["answerable"] is True


def test_parse_response_rejects_non_json():
    with pytest.raises(ValueError, match="non-JSON"):
        _parse_response("sorry, I cannot answer that")


def test_parse_response_rejects_missing_answerable_key():
    with pytest.raises(ValueError, match="answerable"):
        _parse_response('{"answer": "x"}')


def test_parse_response_rejects_non_boolean_answerable():
    """Si le modèle répond {"answerable": "false"} (string) au lieu du booléen JSON, on doit lever une
    erreur explicite plutôt que de laisser `not "false"` (truthy) faire croire qu'il a répondu "true"."""
    with pytest.raises(ValueError, match="booléen"):
        _parse_response('{"answerable": "false"}')


def test_generate_answer_maps_labels_back_to_real_citation_keys():
    passages = [
        ("fastapi/tutorial/background-tasks.md#Using `BackgroundTasks`", "declare a BackgroundTasks param"),
        ("fastapi/tutorial/background-tasks.md#Add the background task", "call add_task"),
    ]
    client = FakeLLMClient({
        "background tasks": '{"answerable": true, "answer": "Use BackgroundTasks.", "sources": ["S1", "S2"]}',
    })
    result = generate_answer(client, "how do background tasks work", passages)
    assert result == {
        "answerable": True,
        "answer": "Use BackgroundTasks.",
        "citations": [
            "fastapi/tutorial/background-tasks.md#Using `BackgroundTasks`",
            "fastapi/tutorial/background-tasks.md#Add the background task",
        ],
    }


def test_generate_answer_refusal_has_no_citations():
    client = FakeLLMClient({"capital of France": '{"answerable": false}'})
    result = generate_answer(client, "what is the capital of France", [("some/doc.md#intro", "irrelevant text")])
    assert result == {"answerable": False, "answer": None, "citations": []}


def test_generate_answer_ignores_hallucinated_source_labels():
    """Le modèle cite un label qui n'existe pas (ex. S9 alors qu'il n'y a que 2 passages) : on l'ignore
    plutôt que de planter, pour ne jamais renvoyer une citation qui ne pointe vers rien de réel."""
    passages = [("some/doc.md#intro", "text")]
    client = FakeLLMClient({"question": '{"answerable": true, "answer": "y", "sources": ["S1", "S9"]}'})
    result = generate_answer(client, "a question", passages)
    assert result["citations"] == ["some/doc.md#intro"]


def test_fake_llm_client_records_calls():
    client = FakeLLMClient({"foo": '{"answerable": false}'})
    generate_answer(client, "foo bar", [("k", "v")])
    assert len(client.calls) == 1
    system, user = client.calls[0]
    assert "foo bar" in user
    assert "answerable" in system


def test_fake_llm_client_raises_on_unscripted_prompt():
    client = FakeLLMClient({"only this": '{"answerable": false}'})
    with pytest.raises(KeyError):
        generate_answer(client, "totally different question", [("k", "v")])
