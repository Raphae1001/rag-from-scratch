# RAG from First Principles

Système RAG construit de bout en bout : BM25 from scratch, recherche dense (pgvector),
fusion hybride, génération avec garde-fous, évaluation, API + Langfuse, fine-tuning LoRA.

**Corpus :** documentation officielle de FastAPI + Starlette + Pydantic (le stack que j'utilise au quotidien), découpée par section (## / ###) : ~1600 documents. Changelogs, pages « méta » et stubs d'API (`pydantic/docs/api/`) exclus, voir `scripts/build_corpus.py`.

## Démarrage
```bash
bash scripts/fetch_docs.sh          # clone les 3 docs à des commits figés (corpus reproductible)
python scripts/build_corpus.py      # -> data/corpus.jsonl (1616 documents)
pip install -r requirements-dev.txt
pytest                              # 147 tests (les tests de base utilisent Docker si `pgserver` est absent)
python scripts/phase1_check.py      # rebuild de l'index + 5 requêtes de contrôle
```
Commits figés : FastAPI `a3d205b`, Starlette `63c5760`, Pydantic `bb6da4c` (voir `scripts/fetch_docs.sh`).

## Avancement
- [x] Phase 1 — Recherche classique (J1–J4)
- [x] Phase 2 — Recherche dense + fusion hybride (recall@10 : BM25 0,576 → hybride 0,739, gain significatif — IC95% bootstrap [0,087, 0,250])
- [x] Phase 3 — Génération + garde-fous (0/12 hallucination, citations vérifiées, reranking mitigé — voir section)
- [x] Phase 4 — Évaluation (fidélité 0,957-0,962 calibrée sur deux runs indépendants — voir section)
- [x] Phase 5 — Prod & observabilité (API Docker, traçage Langfuse vérifié, latence mesurée — voir section)
- [x] Phase 6 — Fine-tuning contrastif (recall@10 dense +0,068 après LoRA — voir section)

## Validation Phase 1 (gate)
| Critère de la spec | Preuve |
|---|---|
| Corpus ≥ 1000 documents | `test_corpus_has_at_least_1000_documents` (1616 docs, 3 sources) |
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
(terme, document) distincts, plus O(N) pour les longueurs. Mesuré sur le corpus (1616 docs,
252 383 tokens, 6446 termes, 118 285 postings) : **< 100 ms**.

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
bash scripts/run_phase2.sh           # tout en une commande (venv, tests, Docker, embeddings, benchmark)
```
ou pas à pas :
```bash
pip install -r requirements-dev.txt
docker compose up -d --wait          # Postgres + pgvector sur localhost:5433
python scripts/embed_corpus.py       # ~2750 passages -> table `chunks` (quelques minutes sur CPU)
python scripts/benchmark.py          # -> results/phase2_benchmark.md et .json
```
Tests de base de données (`tests/test_db.py`) : ils utilisent `pgserver` (Postgres jetable, paquet pip) s'il est installé, sinon le
Postgres de `docker compose` dans une **base dédiée** `rag_test_pytest`, créée puis supprimée ; un garde-fou refuse de vider
toute autre base, donc les embeddings de la base `rag` ne sont jamais touchés.

Sans Docker (dev) : ajouter `--pgdata data/pgdata` aux deux scripts (Postgres+pgvector local via le paquet pip `pgserver`).
**`pgserver` ne publie pas de wheel pour Python ≥ 3.13** (vérifié sur PyPI, builds jusqu'à 3.12 seulement) : sur
Python 3.13+, `pip` l'ignore silencieusement (`requirements-dev.txt`) et seul Docker reste disponible.

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

### Résultats (mesurés : `all-MiniLM-L6-v2`, 44 requêtes, 1616 documents, image Docker `pgvector/pgvector:pg16`)

| Configuration | recall@10 | recall@5 | precision@10 | MRR | latence moy. |
|---|---|---|---|---|---|
| BM25 seul | 0,576 | 0,379 | 0,109 | 0,357 | 0,7 ms |
| Dense seul | 0,617 | 0,432 | 0,120 | 0,412 | 21,9 ms |
| **Hybride (RRF)** | **0,739** | **0,485** | **0,141** | **0,508** | 20,0 ms |

recall@10 par type de requête :

| Configuration | mots-clés (n=16) | reformulations (n=28) |
|---|---|---|
| BM25 seul | 0,844 | 0,423 |
| Dense seul | 0,906 | 0,452 |
| Hybride (RRF) | 1,000 | 0,589 |

**Gate Phase 2 atteint** : recall@10 hybride (0,739) > BM25 seul (0,576), soit +0,163 (+28 % relatif).

Lecture des résultats :
- Le gain vient surtout des **reformulations** (0,423 → 0,589) : c'est le cas visé (« validate » vs « validation », requêtes
  sans les mots de la doc). Sur les requêtes **mots-clés**, l'hybride atteint 1,000 (10/10 points en jeu retrouvés).
- **Le dense seul a légèrement reculé** (0,659 → 0,617) après la correction du corpus décrite ci-dessous, qui ajoute du vrai
  code Python dans ~23 % des documents FastAPI. Hypothèse non vérifiée : `all-MiniLM-L6-v2` est entraîné sur du langage
  naturel, pas du code — des passages plus longs mélangeant prose et code diluent peut-être le signal sémantique, ou le code
  pousse une partie de la prose hors de la fenêtre de 120 mots. La fusion RRF absorbe cette perte (BM25 profite du
  vocabulaire de code ajouté) : le hybride progresse malgré tout (0,720 → 0,739). Non creusé plus loin — signalé tel quel.
- **Latence** : BM25 0,7 ms, dense 21,9 ms, hybride 20,0 ms. Le coût du dense est dominé par l'encodage de la requête sur
  CPU (~20 ms), pas par la recherche SQL. La première exécution du benchmark (avant la correction du corpus) avait affiché
  108,7 ms pour le dense une fois, jamais reproduit depuis (démarrage à froid probable, cause non isolée). Ces latences
  restent **indicatives** (une exécution, 44 requêtes, machine de développement) : une mesure propre demanderait un
  échauffement préalable et plusieurs répétitions (prévu en Phase 5).

Compromis exactitude / vitesse (recherche approximative HNSW, m=16, ef_construction=64, construction 0,55 s ; 2757 passages) :

| Mode | recall@10 vs exact | latence moy. | p95 |
|---|---|---|---|
| exact (sans index) | 1,000 | 4,10 ms | 4,74 ms |
| HNSW ef_search=10 | 0,948 | 0,61 ms | 0,90 ms |
| HNSW ef_search=40 | 0,995 | 0,76 ms | 1,03 ms |
| HNSW ef_search=100 | 1,000 | 0,98 ms | 1,32 ms |

`ef_search=40` (recall 0,995 pour ~5× moins de latence que l'exact) est un bon compromis à cette échelle. À 2750 passages
l'exact reste utilisable (4,1 ms) : l'index HNSW ne devient nécessaire que pour des corpus bien plus grands.

**Intervalle de confiance (bootstrap, `scripts/bootstrap_ci.py`)** : 44 requêtes rééchantillonnées avec remise,
10 000 tirages. Écart hybride − BM25 = **0,163, IC95% [0,087, 0,250]** — l'intervalle exclut 0, le gain est
**statistiquement significatif** sur ce set, malgré sa petite taille. Détail dans `results/phase2_bootstrap_ci.md`.

**Limites de cette mesure** (à garder en tête avant de citer ces chiffres) :
- **44 requêtes seulement** : le bootstrap ci-dessus quantifie l'incertitude due à la taille de l'échantillon,
  mais ne corrige pas un biais dans les annotations elles-mêmes (point suivant).
- **Annotations faites par l'auteur** (et un assistant), à partir du plan des pages et sans regarder les résultats de BM25 ; elles
  sont discutables, et le set est équilibré à la main entre mots-clés (16) et reformulations (28), ce qui influence l'ampleur du gain.
- Paramètres non réglés : k=60 pour RRF, 50 candidats par méthode, passages de 120 mots. Aucun réglage n'a été fait sur ce set,
  donc pas de sur-ajustement, mais aussi pas d'optimisation.
- Le rapport complet est dans `results/phase2_benchmark.md` et `.json`.

### Correction de corpus (post-Phase 4) : exemples de code manquants

En creusant un refus de réponse en Phase 4 (voir plus bas), j'ai trouvé que **372 documents sur 1597 (23,3 % du corpus,
37,6 % des documents FastAPI)** contenaient une directive d'inclusion de code propre au générateur de doc FastAPI
(`{* ../../docs_src/chemin.py ln[a:b] *}`) **jamais résolue** : `scripts/fetch_docs.sh` ne clonait que `docs/en/docs`, pas
le dossier `docs_src/` du dépôt que ces directives référencent. Résultat : environ un quart des documents FastAPI avaient
une explication textuelle correcte mais **sans l'exemple de code réel** — un vrai trou de qualité de corpus, pas une
simple imperfection de présentation.

**Corrigé à la racine** : `fetch_docs.sh` clone maintenant aussi `docs_src/` ; `build_corpus.py::resolve_code_snippets`
parse la directive et insère le vrai code (testé dans `tests/test_build_corpus.py`, y compris une variante de syntaxe
`hl[...] title[...]` découverte en cours de route). Après correction : **1616 documents** (+19, certaines sections
passent le seuil de 15 mots grâce au code ajouté), **0 directive résiduelle non résolue** sur tout le corpus — la
dernière (`fastapi/how-to/configure-swagger-ui.md`, qui référençait le code source de la librairie FastAPI elle-même,
pas `docs_src/`) a été résolue en généralisant le résolveur : il n'est plus ancré sur `docs_src/` spécifiquement, mais
résout tout chemin relatif à la racine du dépôt (`fetch_docs.sh` clone aussi `fastapi/openapi/` en conséquence).
L'ensemble des Phases 2, 3 et 4 a été re-mesuré après chacune de ces deux corrections ; les chiffres de ce README
sont ceux d'après la correction finale.

> Les exécutions `--fake` (embeddings factices, sans sémantique) ne servent qu'à tester la plomberie ; leurs
> résultats sont écrits dans `results/*_FAKE.*` (ignorés par git) et ne doivent jamais être reportés.

## Phase 3 — Génération avec garde-fous

**Ce qui est en place** : reranking par cross-encoder (`rerank.py`), génération sourcée via l'API Claude
(`generate.py`), pipeline complet `scripts/answer.py`, vérification des gates `scripts/phase3_check.py`,
set de 12 requêtes hors-corpus (`data/eval/no_answer.json`).

### Design
- **LLM** : API Claude (`claude-haiku-4-5-20251001`), clé lue depuis `$ANTHROPIC_API_KEY` (jamais dans le code).
- **Reranking** : `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers, local, pas d'appel API) sur les
  candidats du retrieval hybride, avant génération.
- **Garde-fou anti-hallucination** : le modèle reçoit les passages étiquetés `[S1]`, `[S2]`, ... et doit répondre
  en **JSON strict** — `{"answerable": false}` si le contexte ne suffit pas, sinon
  `{"answerable": true, "answer": "...", "sources": ["S1", "S3"]}`. Passer par un format structuré plutôt que de
  chercher « je ne sais pas » dans du texte libre rend la détection déterministe : pas de risque qu'une réponse
  hallucinée commence par une formule de politesse qui échapperait à un test de correspondance de texte.
- **Citations** : le modèle ne voit jamais les vraies clés `chemin#section` (seulement `S1`, `S2`...) — elles
  sont réinjectées après coup par `generate_answer`, pour éviter qu'il invente une clé plausible mais fausse.
  Un label cité qui ne correspond à aucun passage fourni est silencieusement ignoré plutôt que de planter.
- **Testabilité sans réseau** : `LLMClient`/`Reranker` sont des `Protocol` (même pattern que `Embedder` en
  Phase 2) — `FakeLLMClient` et `OverlapReranker` permettent de tester toute la logique de parsing, de mapping
  des citations et de tri sans appeler l'API ni télécharger de modèle (`tests/test_generate.py`, `tests/test_rerank.py`).

### Lancer
```bash
python scripts/answer.py "how do I upload a file to the server"    # une question
python scripts/phase3_check.py                                     # gates -> results/phase3_report.{md,json}
```

### Résultats (mesurés : `claude-haiku-4-5-20251001`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `all-MiniLM-L6-v2`)

**Gate non-hallucination — atteint** : 12/12 requêtes du set de non-réponse (hors du domaine FastAPI/Starlette/
Pydantic — OAuth2 Rails, pandas, Kubernetes, React, Go, « capitale de la France », etc.) correctement identifiées
comme non-répondables, 0 hallucination. Détail dans `results/phase3_report.md`.

**Citations — vérifiées** : chaque réponse répondable cite ses sources réelles (`chemin#section`), par ex. pour
« how do I upload a file to the server » : 5 sources dans `fastapi/tutorial/request-files.md` et
`fastapi/reference/uploadfile.md`, cohérentes avec le contenu de la réponse.

**Gate reranking — atteint, mais résultat mitigé** (44 requêtes annotées, candidats = 30) :

| | recall@10 | MRR |
|---|---|---|
| Hybride seul | 0,739 | 0,508 |
| Hybride + reranking | 0,705 | 0,556 |

Le reranking **améliore le MRR** (+0,048 : le premier résultat pertinent remonte davantage) mais **dégrade le
recall@10** (−0,034 : certains documents pertinents sortent du top-10 après reranking) — même schéma qu'avant la
correction du corpus (voir Phase 2), ce n'est donc pas un artefact du bug de code manquant. Le script accepte le
gate si recall@10 **ou** MRR s'améliore (la spec demande une amélioration mesurée, sans préciser laquelle des
deux métriques) — je le documente ici sans l'enjoliver : le cross-encoder est meilleur pour remonter *la*
bonne réponse en position 1, pas pour préserver toute la couverture du top-10. Une piste non explorée : reranker
sur un pool de candidats plus large que 30 pour voir si le recall@10 se maintient mieux.

`scripts/phase3_check.py`, lancé avec Postgres + embeddings + `ANTHROPIC_API_KEY`, écrit ces chiffres dans
`results/phase3_report.{md,json}`.

## Phase 4 — Évaluation rigoureuse

**Ce qui est en place** : mesure de fidélité (faithfulness) par juge LLM (`faithfulness.py`), vérification
`scripts/phase4_check.py`. Le tableau comparatif des 3 configurations (precision@k/recall@k/MRR) n'est **pas
recalculé** : il est repris tel quel de `results/phase2_benchmark.json`, déjà mesuré et versionné en Phase 2.

### Design
- **Fidélité, pas exactitude** : le juge ne compare pas la réponse à une "bonne réponse" attendue — il vérifie
  seulement que chaque affirmation de la réponse est soutenue par les passages **cités**. Une réponse peut être
  fidèle (rien d'inventé) sans être complète, et inversement ; la spec demande la fidélité au contexte, pas la
  justesse de fond.
- **Second appel LLM indépendant** : plutôt que de comparer des chaînes de caractères (une affirmation vraie
  peut être une reformulation, pas un extrait mot pour mot), on redemande à Claude de découper la réponse en
  affirmations et de juger chacune — même pattern JSON structuré que la Phase 3, avec le même parsing tolérant
  au texte en trop après le JSON (`extract_json_object`, factorisé depuis `generate.py`).
- **Parallélisation** : comme en Phase 3, le retrieval/reranking reste séquentiel (connexion Postgres partagée),
  mais les appels de génération puis de jugement sont chacun parallélisés avec un `ThreadPoolExecutor`.

### Lancer
```bash
python scripts/phase4_check.py    # -> results/phase4_report.{md,json}
```

### Résultats (mesurés : `claude-haiku-4-5-20251001` juge, pipeline Phase 3 complet, 44 requêtes annotées)

**Tableau comparatif final** (repris de la Phase 2) :

| Configuration | recall@10 | recall@5 | precision@10 | MRR |
|---|---|---|---|---|
| BM25 seul | 0,576 | 0,379 | 0,109 | 0,357 |
| Dense seul | 0,617 | 0,432 | 0,120 | 0,412 |
| **Hybride (RRF)** | **0,739** | **0,485** | **0,141** | **0,508** |

**Configuration gagnante : hybride (RRF)** — recall@10 supérieur de +0,163 à BM25 seul, sans coût de latence
supplémentaire notable par rapport au dense seul.

**Fidélité (faithfulness)** : sur les 44 requêtes, **42 répondables, 2 correctement refusées, 0 erreur de
format** (le prompt a été retravaillé en cours de route — voir "Bug rencontré" ci-dessous). Score de fidélité
brut (jugé par LLM, 392 affirmations au total) : **0,949**. Détail dans `results/phase4_report.md`.

**Calibration manuelle du juge, sur deux runs indépendants** (chaque affirmation flaggée relue contre le vrai
texte des passages cités, pas juste survolée) : run 1 (avant la dernière correction de corpus) = 16/392
flaggées, run 2 (après) = 18/392 flaggées. **Sur les deux runs, verdict identique** : la quasi-totalité des
affirmations flaggées sont de vrais écarts de grounding strict, mais **aucune des 34 affirmations flaggées au
total n'est factuellement fausse** — ce sont des faits réels sur FastAPI/Pydantic (ex. « 403 = Forbidden »,
les 4 modes de field validator Pydantic, l'option `--workers`) que le modèle connaît de manière générale mais
qui ne sont pas explicitement dans le **seul passage cité** pour cette réponse précise. C'est un manquement
au grounding strict, pas une hallucination dangereuse. Le même cas précis (`q35`, paramètres Swagger UI) est
une **vraie erreur du juge sur les deux runs** : le passage cité contient explicitement le mécanisme décrit.
**Score corrigé : 0,962 (run 1) et 0,957 (run 2)** — stable à ~0,01 près malgré la non-déterminisme du juge
LLM d'un run à l'autre sur *quelles* affirmations précises il flag. Détail dans `results/phase4_report.json`,
clé `faithfulness.manual_review`.

**Les 2 refus ont été vérifiés un par un** (pas juste comptés) :
- `q16` (« give a model attribute a fallback value when it is missing ») : **vrai raté de retrieval** — le bon
  document (`pydantic/concepts/fields.md#Default values`) n'apparaît jamais dans le top-5 présenté au modèle ;
  à la place, une ambiguïté lexicale fait remonter des pages sur les fallback pages FastAPI et les erreurs de
  validation Pydantic ("missing"), sans rapport avec la question. Cohérent avec la limite déjà documentée en
  Phase 1 (pas de synonymes) — pas corrigé au cas par cas pour ne pas sur-ajuster au set d'éval.
- `q05` (« share one database session across all my endpoints ») : **refus correct et défendable**, pas un bug.
  Le bon document est bien dans le contexte, mais le modèle a noté — à raison — que la doc décrit l'inverse de
  la question littérale : une **nouvelle** session **par requête** via une dependency, pas une session unique
  **partagée**. Il a préféré signaler l'écart plutôt que de deviner l'intention. Je n'ai pas assoupli le prompt
  pour forcer une réponse ici : ça affaiblirait le garde-fou sur des cas réellement ambigus ailleurs.
- Une 3e requête (`q03`, « change how validation errors are returned ») était refusée avant la correction du
  corpus décrite plus haut, et répond maintenant correctement — c'était un vrai raté de retrieval, corrigé
  indirectement en réparant le corpus (le bon document contient maintenant plus de contenu utile).

**Bug rencontré en cours de mesure** : sur la requête `"field validator"` (style mot-clé, très courte), Claude
a répondu par une question de clarification en texte libre au lieu du JSON attendu, faisant planter le script.
Corrigé en deux temps : le prompt système précise maintenant explicitement que ces requêtes sont des recherches
documentaires, pas des questions conversationnelles ambiguës (`SYSTEM_PROMPT` dans `generate.py`) ; et
`_common.py::safe_generate_answer` rend les scripts de mesure résilients à un cas isolé mal formé (compté à
part, jamais confondu avec un refus correct ou une hallucination). Après ce correctif : 0/44 erreurs de format.

### Gates de la spec
- Le set d'évaluation est versionné (`data/eval/queries.json`, 44 requêtes) — déjà vrai depuis la Phase 2
- Chaque métrique est calculée par un script reproductible — `scripts/benchmark.py` (Phase 2) + `phase4_check.py`
- Le tableau comparatif final montre clairement la configuration gagnante avec une explication — ci-dessus

## Phase 5 — Production & observabilité

**Ce qui est en place** : API FastAPI (`src/rag/api.py`, `POST /query` + `GET /health`), `Dockerfile` +
service `api` dans `docker-compose.yml`, traçage Langfuse optionnel, script de latence propre
(`scripts/measure_latency.py`).

### Design
- **Chargement une fois, pas par requête** : le `lifespan` FastAPI charge le corpus, l'index BM25,
  l'embedder, le reranker, la connexion DB et le client Claude au démarrage — le chargement des modèles
  domine largement la latence d'une requête isolée (mesuré en Phase 3), donc on ne le paie qu'une fois.
- **Démarrage sans intervention manuelle** : `docker-entrypoint.sh` appelle `scripts/ensure_embedded.py`
  avant de lancer l'API — si les embeddings ne sont pas déjà chargés (compte de documents ≠ taille du
  corpus), il lance `embed_corpus.py` tout seul. `docker compose up` répond donc à une requête de bout en
  bout sans étape manuelle, y compris sur un volume Postgres vierge (vérifié).
- **Traçage Langfuse optionnel** : absent si `$LANGFUSE_PUBLIC_KEY` n'est pas définie (même principe que
  `pgserver` : une fonctionnalité en moins, pas un plantage). Quand il est configuré, chaque requête crée
  une trace avec 3 observations enfants (`retrieval`, `reranking`, `generation`), chacune avec la vraie
  latence mesurée en métadonnée — ces observations sont créées après coup, une fois le pipeline terminé
  (leur propre durée dans Langfuse est donc quasi nulle), c'est la donnée en métadonnée qui compte, pas la
  durée de l'observation elle-même.
- **Coût par requête** : `AnthropicClient` expose `last_usage` (tokens d'entrée/sortie du dernier appel,
  canal latéral, sans toucher au `Protocol` `LLMClient`) — Langfuse calcule le coût automatiquement à
  partir de ça et du nom du modèle.

### Lancer
```bash
docker compose up -d --wait                       # DB + API ; charge les embeddings si absents
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "how do I upload a file to the server"}'
python scripts/measure_latency.py                 # latence propre -> results/phase5_latency.md
```

### Résultats (mesurés : conteneur Docker, `claude-haiku-4-5-20251001`, Langfuse Cloud)

**Gate Docker — atteint** : `docker compose up -d --wait` démarre les deux conteneurs sur un volume
Postgres **vierge** (testé avec un port différent pour éviter toute collision) et répond à une requête
complète sans aucune étape manuelle — `ensure_embedded.py` a chargé les embeddings tout seul.

**Gate Langfuse — atteint, vérifié visuellement dans le vrai dashboard** (pas juste supposé) : la trace
`69877a6bbd25999f9f5edcca8a615c8b` montre les 4 observations attendues (`query` → `retrieval`,
`reranking`, `generation`), latence par étape correcte en métadonnées (174 ms / 1385 ms / 5849 ms,
identique à la réponse JSON de l'API), et le nœud `generation` affiche le modèle
(`claude-haiku-4-5-20251001`), **1831 tokens** et un coût calculé automatiquement (**$0,003163**).

**Gate latence — mesuré avec échauffement + répétitions** (3 requêtes d'échauffement, 3×5 = 15 mesures,
contre le conteneur Docker) :

| Étape | moyenne (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|
| total (HTTP) | 3712,2 | 3924,6 | 4751,5 |
| retrieval | 39,7 | 40,3 | 52,9 |
| reranking | 975,3 | 968,9 | 1087,9 |
| génération | 2557,9 | 2773,3 | 3619,6 |

La génération domine (appel réseau à l'API Claude, hors de notre contrôle). Le reranking est notablement
plus lent que mesuré en Phase 3 hors Docker (~250 ms) — hypothèse non vérifiée : CPU alloué au conteneur
Docker Desktop plus restreint que sur l'hôte directement. Signalé honnêtement, pas creusé plus loin (15
mesures seulement, pas de quoi trancher une cause précise). Détail dans `results/phase5_latency.md`.

### Limites connues de la Phase 5
- Image Docker lourde (~2 Go, torch inclus) — accepté, même compromis que documenté depuis la Phase 2.
- Pas de retry/backoff sur l'appel à l'API Claude : une erreur réseau fait échouer la requête entière.
- Pas d'authentification sur l'API (`/query` est ouvert) — hors scope pour ce projet d'apprentissage.
- La mesure de latence ci-dessus est sur 15 requêtes seulement (même limite méthodologique que le reste
  du projet : peu de données, pas d'intervalle de confiance).

## Phase 6 — Fine-tuning contrastif

**Ce qui est en place** : génération de paires d'entraînement sans fuite (`scripts/make_training_pairs.py`),
fine-tuning LoRA contrastif (`scripts/finetune_embeddings.py`), ré-évaluation complète style Phase 4
(`scripts/phase6_check.py`), checkpoint versionné (`models/finetuned-minilm-lora/`, 1,3 Mo).

### Design
- **Paires (requête, document) sans LLM, sans fuite** : le titre de page + section sert de "requête" faible,
  le corps du document (sans le titre répété) sert de positif — supervision faible standard pour des
  embeddings de retrieval, gratuite et déterministe (pas d'appel API). **Tous les documents pertinents pour
  le set d'évaluation des 44 requêtes sont exclus de l'échantillon d'entraînement** (82 documents exclus sur
  1616) : sans cette exclusion, le fine-tuning apprendrait spécifiquement à bien retrouver les documents sur
  lesquels on le réévalue ensuite — une fuite qui gonflerait artificiellement le gain mesuré, exactement le
  genre de biais que la règle du projet ("ne jamais ajuster pour faire passer un critère") interdit. 400
  paires échantillonnées sur 1352 éligibles.
- **LoRA, pas un fine-tuning complet** : adaptateurs de rang 16 sur les projections `query`/`value` de
  l'attention (`peft.LoraConfig`), modèle de base gelé — **0,65% des paramètres entraînables** (147k / 22,9M).
  Rapide (18s sur CPU pour 400 paires × 3 époques), peu de mémoire, et le résultat reste un modèle
  `sentence-transformers` standard, rechargeable sans code spécifique.
- **Loss contrastive `MultipleNegativesRankingLoss`** : chaque paire du batch sert de négatif implicite aux
  autres paires du même batch — pas besoin d'annoter des négatifs explicites.
- **Ré-évaluation en mémoire, pas de seconde base Postgres** : le modèle fine-tuné produit des vecteurs
  différents du modèle de base, donc la table `chunks` existante (embeddings du modèle de base) n'est pas
  réutilisable telle quelle. Plutôt que de réindexer dans Postgres, `phase6_check.py` recalcule la recherche
  dense en numpy pur (~2750 passages, largement assez petit pour ne pas avoir besoin d'un index) — même
  méthode, mêmes 44 requêtes, seule la source des vecteurs change.

### Lancer
```bash
python scripts/make_training_pairs.py      # -> data/train/pairs.jsonl (déjà versionné, pas la peine de refaire)
python scripts/finetune_embeddings.py      # -> models/finetuned-minilm-lora/ + results/phase6_loss_curve.png
python scripts/phase6_check.py             # -> results/phase6_report.md
```

### Résultats (mesurés : CPU, 400 paires, 3 époques, 44 requêtes annotées)

**Entraînement** : bout en bout en **18,2s sur CPU** (pas de GPU utilisé ni nécessaire à cette échelle),
147 456 / 22 860 672 paramètres entraînables. Courbe de loss (`results/phase6_loss_curve.png`) : tendance
baissière nette (0,76 → 0,57) mais bruitée — attendu avec seulement 39 pas d'optimisation au total (peu de
données, 3 époques).

| | recall@10 | precision@10 | MRR |
|---|---|---|---|
| Dense (base, `all-MiniLM-L6-v2`) | 0,617 | 0,120 | 0,412 |
| **Dense (fine-tuné LoRA)** | **0,686** | 0,136 | 0,492 |
| Hybride (base) | 0,739 | 0,141 | 0,508 |
| **Hybride (fine-tuné)** | **0,735** | 0,141 | 0,514 |

**Gate Phase 6 atteint** : recall@10 dense +0,068 (+11 % relatif) après fine-tuning. **Mais l'hybride, lui,
ne bouge quasiment pas** (−0,004, dans le bruit) — et c'est expliqué, pas juste rapporté tel quel : la
fusion RRF combine déjà BM25 et dense, donc l'essentiel du gain du dense fine-tuné est **recouvert** par ce
que BM25 trouvait déjà. Le fine-tuning améliore clairement le signal sémantique pur, mais son effet est
dilué une fois fusionné avec un lexical déjà solide. Sur un corpus où BM25 serait plus faible, l'effet du
fine-tuning sur l'hybride serait probablement plus visible.

### Limites connues de la Phase 6
- **400 paires de supervision faible**, pas des requêtes naturelles annotées à la main comme le set d'éval
  — le "signal" d'entraînement (titre → corps) est un proxy, pas une vraie requête utilisateur.
- **Un seul run** : pas de moyenne sur plusieurs seeds/répétitions d'entraînement, donc le gain de +0,068
  pourrait varier avec une autre initialisation. Cohérent avec la limite déjà assumée ailleurs (44 requêtes,
  pas d'intervalle de confiance sur CE chiffre précis — contrairement au gain Phase 2 qui, lui, en a un).
- **CPU seulement testé** : le script détecte `cuda` automatiquement mais n'a pas été exécuté sur GPU ici.
- Hyperparamètres (rang LoRA=16, lr=2e-4, 3 époques) choisis raisonnablement mais non balayés (pas de
  recherche d'hyperparamètres) — pas d'optimisation poussée, juste une preuve que le pipeline fonctionne et
  améliore la métrique visée.

## Licences et crédits
Code : licence MIT (`LICENSE`). Le corpus reprend la documentation de FastAPI, Starlette et Pydantic sous leurs licences (MIT / BSD-3-Clause) : voir
[`NOTICE.md`](NOTICE.md) et `third_party_licenses/`. Ce projet n'est pas affilié à ces projets.

## Limites connues
Pas de scale, pas d'agents (hors scope assumé). Une directive d'inclusion de code résiduelle non résolue dans
tout le corpus (`fastapi/how-to/configure-swagger-ui.md`, référence le code source de la librairie FastAPI
elle-même plutôt que `docs_src/` — voir section Phase 2, "Correction de corpus"). Retrieval imparfait sur les
requêtes à forte ambiguïté lexicale (voir Phase 4, `q16`) : limite connue de la Phase 1 (pas de synonymes),
non corrigée au cas par cas pour ne pas sur-ajuster au set d'évaluation.
