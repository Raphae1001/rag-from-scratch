"""Tokenizer pour BM25 (J1).

Choix de conception :
- minuscules
- on découpe sur tout ce qui n'est pas une lettre (Unicode, donc "café" reste entier)
  ou un chiffre : la ponctuation, les emojis ET l'underscore sont des séparateurs.
  `response_model` -> ["response", "model"] : un utilisateur qui tape "response model"
  doit retrouver la section sur `response_model`.
- les chiffres sont conservés (404, 422, 3 dans "python 3").
- stopwords optionnels : liste courte écrite à la main. On la garde optionnelle car
  BM25 gère déjà les mots courants via l'IDF (leur poids tend vers ~0).
"""
import re

_TOKEN = re.compile(r"[^\W_]+")  # lettres (Unicode) et chiffres ; `_` exclu -> séparateur

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have i if in into is it its of on
    or so that the their then there these they this to was were will with you your
    """.split()
)


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
