# Intervalle de confiance bootstrap — recall@10 hybride vs BM25 (44 requêtes, 10000 rééchantillonnages)

| | moyenne | IC 95% |
|---|---|---|
| BM25 seul | 0.576 | [0.455, 0.689] |
| Hybride (RRF) | 0.739 | [0.633, 0.833] |
| **Écart (hybride − BM25)** | **0.163** | **[0.087, 0.250]** |

**L'IC95% de l'écart exclut 0 : oui** — le gain de l'hybride sur BM25 est statistiquement significatif à ce niveau de confiance, sur ces 44 requêtes.

Méthode : bootstrap non paramétrique — 44 requêtes rééchantillonnées avec remise à chaque tirage, recall@10 moyen recalculé, IC = percentiles 2,5/97,5 de la distribution des moyennes rééchantillonnées. Ne corrige pas le biais d'échantillonnage des annotations elles-mêmes (voir limites du set d'éval, README section Phase 2) — seulement l'incertitude due à la petite taille de l'échantillon (n=44).
