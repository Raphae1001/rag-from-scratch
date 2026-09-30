# Vérification Phase 6 — fine-tuning contrastif (44 requêtes annotées, 2757 passages)

## recall@10 : modèle de base (Phase 2) vs modèle fine-tuné

| | recall@10 | precision@10 | MRR |
|---|---|---|---|
| Dense (base, all-MiniLM-L6-v2) | 0.617 | 0.120 | 0.412 |
| **Dense (fine-tuné LoRA)** | **0.686** | 0.136 | 0.492 |
| Hybride (base) | 0.739 | 0.141 | 0.508 |
| **Hybride (fine-tuné)** | **0.735** | 0.141 | 0.514 |

**Δ recall@10 dense** : +0.068 | **Δ recall@10 hybride** : -0.004

**Gate Phase 6** — amélioration mesurée (dense ou hybride) : **OK**
