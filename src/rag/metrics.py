"""Métriques de retrieval. `ranked` = liste ordonnée de clés de documents, `relevant` = ensemble de clés."""


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Part des documents pertinents retrouvés dans le top-k."""
    if not relevant:
        raise ValueError("relevant est vide : requête non annotée")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Part du top-k qui est pertinente (le dénominateur est k, même si moins de k résultats)."""
    return len(set(ranked[:k]) & relevant) / k


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / rang du premier document pertinent (0 s'il n'y en a pas). La moyenne sur les requêtes = MRR."""
    for rank, key in enumerate(ranked, 1):
        if key in relevant:
            return 1.0 / rank
    return 0.0
