"""Fusion de classements (BM25 + dense)."""


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion : score(d) = somme sur les classements de 1 / (k + rang(d)), rang à partir de 1.

    Ne dépend que des rangs, pas des scores : c'est ce qui permet de mélanger BM25 (échelle ~0-20)
    et une similarité cosinus (~0-1) sans normalisation. k=60 est la valeur de l'article original ;
    plus k est grand, plus les rangs élevés (1, 2, 3) perdent leur avantage.
    Égalités départagées par doc_id croissant (résultat déterministe).
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
