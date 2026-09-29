"""Reranking des candidats du retrieval hybride par un cross-encoder (query, passage) -> score.

Contrairement à BM25/dense (un vecteur par document, comparé indépendamment de la requête), un
cross-encoder lit la requête ET le passage ensemble : plus coûteux (un forward pass par paire), donc
on ne l'applique qu'aux quelques dizaines de candidats déjà retenus par le retrieval, pas au corpus entier.
"""
from typing import Protocol

from rag.tokenizer import tokenize


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[tuple[int, str]]) -> list[tuple[int, float]]: ...


class CrossEncoderReranker:
    """Modèle réel : cross-encoder/ms-marco-MiniLM-L-6-v2 (téléchargé depuis Hugging Face au premier appel)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # import tardif : lourd (torch)

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[tuple[int, str]]) -> list[tuple[int, float]]:
        """candidates : (doc_id, texte). Renvoie les mêmes doc_id triés par score décroissant."""
        if not candidates:
            return []
        pairs = [[query, text] for _, text in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip((doc_id for doc_id, _ in candidates), (float(s) for s in scores)), key=lambda kv: -kv[1])
        return ranked


class OverlapReranker:
    """Faux reranker DÉTERMINISTE (chevauchement de tokens requête/passage) pour les tests et la démo
    hors ligne. Capture un signal lexical grossier, PAS une pertinence sémantique réelle : ne jamais
    l'utiliser pour mesurer un gate."""

    def rerank(self, query: str, candidates: list[tuple[int, str]]) -> list[tuple[int, float]]:
        q_tokens = set(tokenize(query))
        scored = [(doc_id, float(len(q_tokens & set(tokenize(text))))) for doc_id, text in candidates]
        return sorted(scored, key=lambda kv: (-kv[1], kv[0]))
