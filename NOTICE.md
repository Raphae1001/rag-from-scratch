# Contenus tiers / Third-party notice

Le corpus `data/corpus.jsonl` (et le set d'évaluation `data/eval/queries.json`, qui en référence les sections) est **dérivé de la
documentation officielle** de trois projets open source, découpée en sections. Ce texte appartient à ses auteurs et est
redistribué ici sous leurs licences respectives, dont les textes complets sont dans `third_party_licenses/` :

| Projet | Licence | Copyright | Source (commit figé) |
|---|---|---|---|
| FastAPI | MIT | © 2018 Sebastián Ramírez | https://github.com/fastapi/fastapi @ `a3d205bf19640528718cb4f05ab77f4dfca6ad9a` |
| Starlette | BSD 3-Clause | © 2018, Encode OSS Ltd. | https://github.com/encode/starlette @ `63c5760d8a672cee96e1e523d84bfa1c77d9ee4c` |
| Pydantic | MIT | © 2017 to present Pydantic Services Inc. and individual contributors | https://github.com/pydantic/pydantic @ `bb6da4cfbb1f559885ea2fa207ec93853bfeac64` |

Ce dépôt n'est ni affilié à ces projets ni approuvé par eux. Le corpus est regénérable à l'identique avec
`bash scripts/fetch_docs.sh && python scripts/build_corpus.py`.

Le code de ce dépôt (hors corpus) est sous licence MIT (voir `LICENSE`).
Le modèle d'embeddings `sentence-transformers/all-MiniLM-L6-v2` est téléchargé au premier lancement depuis Hugging Face
(non redistribué ici) ; se reporter à sa fiche pour sa licence.
