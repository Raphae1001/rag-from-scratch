"""Embedders. Un embedder renvoie des vecteurs L2-normalisés (produit scalaire = cosinus)."""
import hashlib
from typing import Protocol

import numpy as np

from rag.tokenizer import tokenize

DIM = 384  # all-MiniLM-L6-v2


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Modèle réel (téléchargé depuis Hugging Face au premier appel)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # import tardif : lourd (torch)

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 200, convert_to_numpy=True).astype(np.float32)


class HashingEmbedder:
    """Faux embedder DÉTERMINISTE (sac de mots haché, 384 dimensions) pour les tests et la démo hors
    ligne. Il capture un recouvrement lexical, PAS de sémantique : ne jamais l'utiliser pour évaluer."""

    def __init__(self, dim: int = DIM):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
