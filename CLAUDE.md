# RAG from First Principles — contexte du projet

Projet d'apprentissage (et d'entretien) de Raphael : un RAG de bout en bout en 6 phases, sur la doc FastAPI + Starlette + Pydantic.
Répondre en **français**. Le détail des phases 3 à 6 est dans les PDF de la spec (`Planning.pdf`, `Spec complète.pdf`) : les mettre dans
`docs/spec/` (ignoré par git, ne pas les publier) et les lire avant de démarrer la phase suivante.

## État
- Phase 1 (BM25 from scratch) : **terminée**. Phase 2 (dense pgvector + fusion RRF) : **terminée**, gate atteint et mesuré
  (recall@10 hybride 0,720 > BM25 0,572, 44 requêtes, `results/phase2_benchmark.md`).
- Phase 3 (génération + garde-fous) : **terminée**, gates mesurés (`results/phase3_report.md`) — 0/12 hallucination
  (parfait), citations vérifiées, reranking mitigé (MRR 0,474→0,521 mieux, recall@10 0,720→0,686 moins bien : le
  cross-encoder remonte mieux la 1re bonne réponse mais fait sortir des docs pertinents du top-10). Documenté
  honnêtement dans le README, pas caché.
- Phase 4 (évaluation) : **terminée**, mesuré (`results/phase4_report.md`) — 41/44 répondables, fidélité 0,964
  (borne basse honnête, juge LLM non calibré à la main), 0 erreur de format après correction du prompt sur les
  requêtes courtes style mot-clé (voir README, section Phase 4, "bug rencontré").
- Phases 5 (API + observabilité), 6 (fine-tuning contrastif) : à faire.

## Commandes
```bash
bash scripts/fetch_docs.sh && python scripts/build_corpus.py   # corpus (commits figés) -> data/corpus.jsonl (1597 docs)
docker compose up -d --wait                                    # Postgres 16 + pgvector, localhost:5433
python scripts/embed_corpus.py && python scripts/benchmark.py  # embeddings + benchmark Phase 2
bash scripts/run_phase2.sh                                     # tout, avec venv
python scripts/answer.py "une question"                        # Phase 3 : pipeline complet (besoin d'ANTHROPIC_API_KEY)
python scripts/phase3_check.py                                 # Phase 3 : gates -> results/phase3_report.md
python scripts/phase4_check.py                                 # Phase 4 : fidélité -> results/phase4_report.md
python -m pytest -q                                            # 121 tests
```

## Règles de travail
- Raphael doit pouvoir **expliquer chaque ligne** (surtout `src/rag/bm25.py`) : expliquer, faire écrire, relire ; pas seulement livrer du code.
- Tests écrits avec (ou avant) le code. Ne **jamais** déclarer un gate atteint sans l'avoir mesuré ; ne jamais ajuster le set d'évaluation
  pour faire passer un critère ; signaler honnêtement les limites (44 requêtes, annotations discutables, latences indicatives).
- Les tests de base de données tournent dans une base dédiée `rag_test_pytest` avec un garde-fou : **ne jamais vider la base `rag`**
  (elle contient les embeddings). Aucun secret dans git (`.env` ignoré ; `.env.example` ne contient que des valeurs de dev locales).
- Le corpus est du contenu tiers (MIT / BSD-3) : conserver `NOTICE.md` et `third_party_licenses/`.

## Décisions déjà prises (ne pas les changer sans raison)
Tokenizer Unicode, `_` = séparateur, sans stemming · IDF variante Lucene, k1=1,2, b=0,75 · passages de 120 mots (chevauchement 20),
score d'un document = meilleur passage · embeddings `all-MiniLM-L6-v2` (384 dim, cosinus) · fusion RRF k=60 sur 50 candidats ·
HNSW m=16, ef_construction=64, ef_search=40 (créé/supprimé par le benchmark) · corpus : changelogs, pages « méta » et `pydantic/docs/api/` exclus.

## En attente (avant toute publication)
1. ~~Réécrire l'historique git local avec l'adresse noreply GitHub de Raphael~~ — fait : historique réécrit
   (`89992240+Raphae1001@users.noreply.github.com`), dépôt privé [Raphae1001/rag-from-scratch](https://github.com/Raphae1001/rag-from-scratch)
   créé et poussé sur `master`. Passage en public plus tard, à sa demande.
2. Mesure de latence propre (échauffement + répétitions) : prévue en Phase 5.
