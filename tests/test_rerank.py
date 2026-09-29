from rag.rerank import OverlapReranker


def test_rerank_orders_by_token_overlap_with_query():
    candidates = [
        (1, "this passage is about background tasks and yield dependencies"),
        (2, "this passage never mentions the query terms at all"),
        (3, "background tasks appear here too, as a set they overlap just as much as passage 1"),
    ]
    ranked = OverlapReranker().rerank("background tasks", candidates)
    # overlap est un ensemble de tokens (pas un compte) : 1 et 3 partagent {background, tasks} à égalité,
    # départagés par doc_id croissant ; 2 n'a aucun chevauchement donc arrive dernier.
    assert [doc_id for doc_id, _ in ranked] == [1, 3, 2]


def test_rerank_ties_broken_by_doc_id():
    candidates = [(5, "no overlap here"), (2, "no overlap here either")]
    ranked = OverlapReranker().rerank("something else entirely", candidates)
    assert [doc_id for doc_id, _ in ranked] == [2, 5]


def test_rerank_empty_candidates():
    assert OverlapReranker().rerank("anything", []) == []
