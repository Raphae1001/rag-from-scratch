"""Index inversé : terme -> {doc_id: fréquence}. (J2)

Structure :
- postings[terme] = {doc_id: f(terme, doc)}   -> permet de ne visiter, à la requête,
  que les documents contenant au moins un terme de la requête.
- doc_len[doc_id] = nombre de tokens du document (pour la normalisation BM25)
- avgdl = longueur moyenne des documents

doc_id = position du document dans la liste passée à build_index.
"""
from collections import Counter
from dataclasses import dataclass, field

from rag.tokenizer import tokenize


@dataclass
class InvertedIndex:
    postings: dict[str, dict[int, int]] = field(default_factory=dict)
    doc_len: list[int] = field(default_factory=list)
    avgdl: float = 0.0

    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    def doc_freq(self, term: str) -> int:
        """n_t : nombre de documents contenant `term`."""
        return len(self.postings.get(term, {}))


def build_index(texts: list[str]) -> InvertedIndex:
    """Construit l'index en un seul passage sur le corpus : O(T), T = nombre total de tokens."""
    postings: dict[str, dict[int, int]] = {}
    doc_len: list[int] = []
    for doc_id, text in enumerate(texts):
        tokens = tokenize(text)
        doc_len.append(len(tokens))
        for term, freq in Counter(tokens).items():
            postings.setdefault(term, {})[doc_id] = freq
    avgdl = sum(doc_len) / len(doc_len) if doc_len else 0.0
    return InvertedIndex(postings=postings, doc_len=doc_len, avgdl=avgdl)
