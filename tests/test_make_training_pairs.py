import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from make_training_pairs import clean_title, make_pairs  # noqa: E402


def mk(path, page, section, body, n_words=20):
    text = f"{page}\n{section}\n{body}"
    return {"path": path, "page": page, "section": section, "text": text, "_body": body}


def test_clean_title_strips_anchor():
    assert clean_title("Advanced Middleware { #advanced-middleware }") == "Advanced Middleware"
    assert clean_title("No Anchor Here") == "No Anchor Here"


def test_make_pairs_excludes_relevant_documents():
    body = " ".join(["word"] * 20)
    docs = [
        mk("a.md", "Page A", "Section A", body),
        mk("b.md", "Page B", "Section B", body),
    ]
    pairs = make_pairs(docs, excluded_keys={"a.md#Section A"})
    assert [p["doc_key"] for p in pairs] == ["b.md#Section B"]


def test_make_pairs_skips_short_bodies():
    docs = [mk("a.md", "Page A", "Section A", "too short")]
    assert make_pairs(docs, excluded_keys=set()) == []


def test_make_pairs_query_is_title_plus_section_not_leaking_body():
    body = " ".join(["word"] * 20)
    docs = [mk("a.md", "Page A { #a }", "Section A", body)]
    pairs = make_pairs(docs, excluded_keys=set())
    assert pairs[0]["query"] == "Page A Section A"
    assert pairs[0]["text"] == body
    assert "Page A" not in pairs[0]["text"]  # le titre n'est pas répété dans le positif
