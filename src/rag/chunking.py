"""Découpage des documents en passages pour l'embedding.

all-MiniLM-L6-v2 tronque à 256 « word pieces » (~120-190 mots selon la part de code). Environ 25 %
des sections du corpus dépassent cette fenêtre : sans découpage, la fin des sections longues serait
invisible pour la recherche dense. On découpe donc en fenêtres de `size` mots avec un chevauchement,
et le score d'un document est celui de son meilleur passage.
"""


def chunk_words(text: str, size: int = 120, overlap: int = 20) -> list[str]:
    if size <= overlap:
        raise ValueError("size doit être > overlap")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def make_chunks(doc: dict, size: int = 120, overlap: int = 20) -> list[str]:
    """Passages d'un document. Les passages après le premier reçoivent « page - section : » en tête,
    car le premier contient déjà le titre de page et de section (voir build_corpus.py)."""
    chunks = chunk_words(doc["text"], size, overlap)
    context = f"{doc['page']} - {doc['section']}: "
    return [c if i == 0 else context + c for i, c in enumerate(chunks)]
