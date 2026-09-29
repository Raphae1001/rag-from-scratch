# RAG from First Principles

Système RAG construit de bout en bout : BM25 from scratch, recherche dense (pgvector),
fusion hybride, génération avec garde-fous, évaluation, API + Langfuse, fine-tuning LoRA.

**Corpus :** documentation officielle de FastAPI + Starlette + Pydantic (le stack que j'utilise au quotidien), découpée par section (## / ###) : ~1600 documents. Changelogs, pages « méta » et stubs d'API (`pydantic/docs/api/`) exclus, voir `scripts/build_corpus.py`.

## Démarrage
```bash
bash scripts/fetch_docs.sh          # clone les 3 docs à des commits figés (corpus reproductible)
python scripts/build_corpus.py      # -> data/corpus.jsonl (1597 documents)
pip install -r requirements.txt
pytest                              # 93 tests
python scripts/phase1_check.py      # rebuild de l'index + 5 requêtes de contrôle
```
Commits figés : FastAPI `a3d205b`, Starlette `63c5760`, Pydantic `bb6da4c` (voir `scripts/fetch_docs.sh`).

## Avancement
- [x] Phase 1 — Recherche classique (J1–J4)
- [~] Phase 2 — Recherche dense (code + tests OK ; **benchmark avec le vrai modèle à exécuter**, voir ci-dessous)
- [ ] Phase 3 — Génération + garde-fous
- [ ] Phase 4 — Évaluation
- [ ] Phase 5 — Prod & observabilité
- [ ] Phase 6 — Fine-tuning contrastif

## Validation Phase 1 (gate)
| Critère de la spec | Preuve |
|---|---|
| Corpus ≥ 1000 documents | `test_corpus_has_at_least_1000_documents` (1597 docs, 3 sources) |
| Index reconstruit sans erreur sur le corpus complet | `python scripts/phase1_check.py` + `test_rebuild_is_deterministic` |
| Requêtes visiblement pertinentes (5 requêtes) | `test_gate_query_returns_expected_page_in_top5` (5 cas) + sortie lisible de `phase1_check.py` |
| Score BM25 calculé à la main = code (non-régression) | `tests/test_bm25.py` (mini-corpus de 3 docs, valeurs dérivées ligne à ligne) |
| BM25 indexé correct sur le vrai corpus | `test_indexed_bm25_matches_naive_full_scan` (comparaison à un BM25 naïf sans index, 8 requêtes × 4 jeux de k1/b) |
| Scores identiques à une bibliothèque de référence | `tests/test_reference_rank_bm25.py` : écart < 1e-9 avec `rank_bm25` (IDF alignée) ; même top-1 et ≥ 8/10 communs avec son IDF native |
| Tests unitaires tokenizer / index / scoring | `tests/test_tokenizer.py`, `test_index.py`, `test_bm25.py` |
| Complexité documentée et justifiée | section ci-dessous |

## Complexité (Phase 1)
### Construction de l'index : O(T) temps, O(P) mémoire
Un seul passage sur le corpus : chaque document est tokenisé (linéaire en sa taille), ses fréquences
sont comptées avec un `Counter`, puis insérées dans `postings[terme][doc_id]` (insertion dict en O(1)
amorti). Coût total O(T), T = nombre total de tokens. La mémoire est O(P), P = nombre de couples
(terme, document) distincts, plus O(N) pour les longueurs. Mesuré sur le corpus (1597 docs,
222 170 tokens, 6320 termes, 110 605 postings) : **< 100 ms**.

### Requête : O(Σ_t |postings(t)| + m log k)
On ne parcourt que les listes de postings des termes de la requête, jamais les N documents :
le coût dépend de la fréquence documentaire des termes, pas de la taille du corpus. Les scores
sont accumulés dans un dict (m = nombre de documents touchés), puis le top-k est extrait avec
`heapq.nsmallest` en O(m log k), plus économe qu'un tri complet en O(m log m) quand k << m.
Le calcul de l'IDF est en O(1) par terme (la longueur d'une liste de postings donne n_t).
Mesuré : **~0,1 ms par requête** en moyenne.

### Pourquoi ces choix
- Index inversé plutôt que scan linéaire : c'est ce qui rend BM25 utilisable à grande échelle.
- IDF variante Lucene `ln(1 + (N - n_t + 0.5)/(n_t + 0.5))` : toujours positif, alors que la
  version originale devient négative pour un terme présent dans plus de la moitié des documents.
- Égalités de score départagées par `doc_id` pour des résultats déterministes et testables.

### Limites connues de la Phase 1
- **Pas de stemming ni de synonymes** : « validate » ne trouve pas « validation ». Choix assumé : c'est la
  faiblesse que la recherche dense et la fusion hybride (Phase 2) doivent corriger, et elle sert de baseline
  honnête pour la Phase 4.
- **Corpus déséquilibré** : FastAPI 62 %, Pydantic 31 %, Starlette 7 % des documents. Et
  `pydantic/errors/validation_errors.md` fournit à lui seul 111 documents (une section par type d'erreur).
- **Découpage par titres `##`/`###`** : la taille des documents est très variable (18 à 1531 tokens) et les
  sections « introduction » n'ont pour contexte que le titre de leur page.
- **Les 5 requêtes de contrôle** du gate sont un filet de non-régression choisi par l'auteur ; la vraie
  mesure de qualité est le set annoté de la Phase 4.

## Phase 2 — Recherche dense et fusion hybride

**Ce qui est en place** : `docker-compose.yml` (Postgres 16 + pgvector), schéma `db/init.sql`, découpage en
passages (`chunking.py`), embeddings (`embed.py`), recherche cosinus + HNSW (`db.py`), fusion RRF (`fusion.py`),
métriques (`metrics.py`), set annoté de 44 requêtes (`data/eval/queries.json`), scripts `embed_corpus.py` et `benchmark.py`.

### Lancer (avec le vrai modèle `all-MiniLM-L6-v2`)
```bash
pip install -r requirements.txt
docker compose up -d                 # Postgres + pgvector sur localhost:5433
python scripts/embed_corpus.py       # ~2500 passages -> table `chunks` (quelques minutes sur CPU)
python scripts/benchmark.py          # -> results/phase2_benchmark.md et .json
```
Sans Docker (dev) : ajouter `--pgdata data/pgdata` aux deux scripts (Postgres+pgvector local via le paquet pip `pgserver`).

### Choix de conception
- **Passages de 120 mots (chevauchement 20)** : le modèle tronque à 256 word pieces et ~25 % des sections dépassent
  cette fenêtre. Le score d'un document = celui de son meilleur passage (`test_chunking_makes_the_tail_...`).
- **Similarité cosinus** sur vecteurs normalisés (`<=>` de pgvector).
- **Fusion RRF** (k=60) sur les 50 premiers de chaque méthode : elle ne dépend que des rangs, donc pas de
  normalisation entre le score BM25 (~0-20) et le cosinus (~0-1).
- **HNSW non créé au démarrage** : `benchmark.py` le crée et le supprime pour comparer exact / approximatif.
  Le mode exact force `enable_seqscan=on` : désactiver seulement l'index ne suffit pas (`exact_search` + test).

### Set d'évaluation (`data/eval/queries.json`)
44 requêtes : 16 « mots-clés » (noms d'API, style développeur pressé) et 28 « reformulations » sans les mots de la
doc. Annotées **sans passer par BM25** (à partir du plan des pages), clés `chemin#section` vérifiées contre le corpus
(`tests/test_eval_set.py`). C'est un **brouillon à relire** : les annotations peuvent être discutées, et le mélange
mots-clés / reformulations conditionne l'ampleur du gain mesuré, d'où la ventilation par type dans le benchmark.

### Résultats
**Mesuré (BM25 seul, ne dépend pas des embeddings)** : recall@10 = **0,572** (mots-clés 0,812 ; reformulations 0,435),
MRR = 0,333. C'est la barre que la fusion hybride doit dépasser.

**À mesurer avec le vrai modèle** : recall@10 dense et hybride, et compromis exact / HNSW → coller ici le tableau de
`results/phase2_benchmark.md`. Gate Phase 2 : recall@10 hybride > BM25 seul. **Non revendiqué tant que non mesuré.**

> Les exécutions `--fake` (embeddings factices, sans sémantique) ne servent qu'à tester la plomberie ; leurs
> résultats sont écrits dans `results/*_FAKE.*` (ignorés par git) et ne doivent jamais être reportés.

## Limites connues
Pas de scale, pas d'agents (hors scope assumé).
