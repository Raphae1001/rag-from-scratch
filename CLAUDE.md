# RAG from First Principles — contexte du projet

Projet d'apprentissage (et d'entretien) de Raphael : un RAG de bout en bout en 6 phases, sur la doc FastAPI + Starlette + Pydantic.
Répondre en **français**. Le détail des phases 3 à 6 est dans les PDF de la spec (`Planning.pdf`, `Spec complète.pdf`) : les mettre dans
`docs/spec/` (ignoré par git, ne pas les publier) et les lire avant de démarrer la phase suivante.

## État
- Phase 1 (BM25 from scratch) : **terminée**. Phase 2 (dense pgvector + fusion RRF) : **terminée**, gate atteint et mesuré
  (recall@10 hybride 0,739 > BM25 0,576, 44 requêtes, `results/phase2_benchmark.md`). Gain **statistiquement
  significatif** : IC95% bootstrap de l'écart = [0,087, 0,250], exclut 0 (`scripts/bootstrap_ci.py`,
  `results/phase2_bootstrap_ci.md`).
- Phase 3 (génération + garde-fous) : **terminée**, gates mesurés (`results/phase3_report.md`) — 0/12 hallucination
  (parfait), citations vérifiées, reranking mitigé (MRR 0,508→0,556 mieux, recall@10 0,739→0,705 moins bien : le
  cross-encoder remonte mieux la 1re bonne réponse mais fait sortir des docs pertinents du top-10). Documenté
  honnêtement dans le README, pas caché.
- Phase 4 (évaluation) : **terminée**, mesuré (`results/phase4_report.md`) — 42/44 répondables, fidélité
  **0,957-0,962 calibrée à la main sur deux runs indépendants** (376→377/392 puis 374→375/392 après relecture
  des affirmations flaggées contre le vrai texte des passages cités : quasi toutes sont de vrais écarts de
  grounding strict mais AUCUNE n'est factuellement fausse sur les deux runs, 1 était systématiquement une erreur
  du juge — même cas les deux fois). Score stable à ~0,01 près malgré la non-déterminisme du juge LLM.
  0 erreur de format après correction du prompt sur les requêtes courtes style mot-clé. Les 2 refus ont été
  vérifiés un par un (pas juste comptés) : `q16` = vrai raté de retrieval (limite Phase 1 connue), `q05` =
  refus correct et défendable (garde-fou qui fonctionne sur une question littéralement ambiguë), voir README
  section Phase 4.
- **Corrections de corpus post-Phase 4** : (1) 23,3% des documents (37,6% des FastAPI) avaient une directive
  `{* docs_src/... *}` de FastAPI non résolue (code manquant) — `fetch_docs.sh` ne clonait pas `docs_src/` ;
  (2) résolveur généralisé pour couvrir aussi une directive référençant le code source de FastAPI lui-même
  (`fastapi/openapi/`), pas seulement `docs_src/`. Corrigé (`build_corpus.py::resolve_code_snippets`, testé
  dans `test_build_corpus.py`) ; corpus 1597→1616 docs, **0 directive non résolue** ; Phases 2/3/4 entièrement
  re-mesurées après chaque correction (chiffres ci-dessus = après correction finale).
- Phase 5 (API + observabilité) : **terminée**, gates vérifiés — Docker démarre sans intervention manuelle
  (`ensure_embedded.py` charge les embeddings seul), traçage Langfuse **vérifié visuellement dans le vrai
  dashboard** (trace avec les 4 observations attendues, coût/tokens/modèle corrects), latence propre mesurée
  avec échauffement (`results/phase5_latency.md`). Reranking plus lent en Docker qu'en natif (975ms vs 250ms) :
  signalé, pas creusé (hypothèse CPU alloué au conteneur, non vérifiée).
- Phase 6 (fine-tuning contrastif) : **terminée**, gate mesuré (`results/phase6_report.md`) — LoRA (rang 16,
  0,65% des paramètres), 400 paires titre→corps sans LLM, **82 documents pertinents pour le set d'éval
  explicitement exclus de l'entraînement** (pas de fuite). Recall@10 dense +0,068 (0,617→0,686), hybride quasi
  stable (−0,004, expliqué : RRF dilue déjà le gain dense avec BM25). 18,2s d'entraînement sur CPU, checkpoint
  versionné dans `models/finetuned-minilm-lora/` (1,3 Mo).
- **Les 6 phases de la spec sont terminées.** Ce qui reste hors scope, assumé (voir README "Limites connues") :
  pas de scale, pas d'auth sur l'API, pas de retry réseau, pas de synonymes BM25, corpus tiers non affilié.

## Commandes
```bash
bash scripts/fetch_docs.sh && python scripts/build_corpus.py   # corpus (commits figés) -> data/corpus.jsonl (1616 docs)
docker compose up -d --wait                                    # DB + API, localhost:5433/8000
python scripts/embed_corpus.py && python scripts/benchmark.py  # embeddings + benchmark Phase 2
bash scripts/run_phase2.sh                                     # tout, avec venv
python scripts/answer.py "une question"                        # Phase 3 : pipeline complet (besoin d'ANTHROPIC_API_KEY)
python scripts/phase3_check.py                                 # Phase 3 : gates -> results/phase3_report.md
python scripts/phase4_check.py                                 # Phase 4 : fidélité -> results/phase4_report.md
curl -X POST localhost:8000/query -d '{"query":"..."}'          # Phase 5 : API (besoin de docker compose up)
python scripts/measure_latency.py                               # Phase 5 : latence -> results/phase5_latency.md
python scripts/bootstrap_ci.py                                  # IC bootstrap Phase 2 -> results/phase2_bootstrap_ci.md
python scripts/make_training_pairs.py && python scripts/finetune_embeddings.py   # Phase 6 : LoRA -> models/finetuned-minilm-lora/
python scripts/phase6_check.py                                  # Phase 6 : gate -> results/phase6_report.md
python -m pytest -q                                             # 147 tests
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
2. ~~Mesure de latence propre (échauffement + répétitions)~~ — fait en Phase 5, voir `results/phase5_latency.md`.
