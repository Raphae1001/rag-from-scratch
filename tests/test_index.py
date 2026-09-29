from rag.index import build_index

TEXTS = [
    "apple banana apple",       # d0, 3 tokens
    "banana cherry",            # d1, 2 tokens
    "apple cherry cherry cherry date",  # d2, 5 tokens
]


def test_doc_len_and_avgdl():
    idx = build_index(TEXTS)
    assert idx.doc_len == [3, 2, 5]
    assert idx.n_docs == 3
    assert idx.avgdl == 10 / 3


def test_postings_hold_frequencies():
    idx = build_index(TEXTS)
    assert idx.postings["apple"] == {0: 2, 2: 1}
    assert idx.postings["cherry"] == {1: 1, 2: 3}
    assert idx.postings["date"] == {2: 1}


def test_doc_freq():
    idx = build_index(TEXTS)
    assert idx.doc_freq("apple") == 2
    assert idx.doc_freq("date") == 1
    assert idx.doc_freq("absent") == 0


def test_empty_corpus():
    idx = build_index([])
    assert idx.n_docs == 0 and idx.avgdl == 0.0 and idx.postings == {}


def test_empty_document_is_counted():
    idx = build_index(["", "hello"])
    assert idx.doc_len == [0, 1]
    assert idx.postings == {"hello": {1: 1}}
