"""Génère les paires (requête, document pertinent) pour le fine-tuning contrastif (Phase 6).

Méthode : le titre de page + section sert de "requête" faible, le corps du document (sans le titre
répété en tête) sert de document positif — technique de supervision faible standard pour entraîner des
embeddings de retrieval (le titre d'une section est une paraphrase naturelle de ce qu'elle contient).

Important : tous les documents "pertinents" du set d'évaluation (`data/eval/queries.json`) sont EXCLUS de
l'échantillon d'entraînement. Sans ça, le fine-tuning apprendrait spécifiquement à bien retrouver les
documents sur lesquels on va ensuite le réévaluer — une fuite qui gonflerait artificiellement le recall@10
mesuré après coup, exactement le genre de biais que la règle "ne jamais ajuster pour faire passer un
critère" du projet interdit.

Usage : python scripts/make_training_pairs.py [--n 300] [--seed 0]
"""
import argparse
import json
import re

from _common import EVAL, ROOT, load_corpus

ANCHOR_SUFFIX = re.compile(r"\s*\{\s*#[^}]*\}\s*$")  # retire l'ancre "{ #id }" du titre de page
OUT = ROOT / "data" / "train" / "pairs.jsonl"


def clean_title(page: str) -> str:
    return ANCHOR_SUFFIX.sub("", page).strip()


def make_pairs(docs: list[dict], excluded_keys: set[str]) -> list[dict]:
    pairs = []
    for d in docs:
        key = f"{d['path']}#{d['section']}"
        if key in excluded_keys:
            continue
        prefix = f"{d['page']}\n{d['section']}\n"
        if not d["text"].startswith(prefix):
            continue  # sécurité : structure inattendue, on saute plutôt que de générer une paire fausse
        body = d["text"][len(prefix):].strip()
        if len(body.split()) < 15:
            continue  # corps trop court pour être un positif informatif
        query = f"{clean_title(d['page'])} {d['section']}".strip()
        pairs.append({"query": query, "doc_key": key, "text": body})
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="nombre de paires à échantillonner")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    docs = load_corpus()
    eval_queries = json.loads(EVAL.read_text(encoding="utf-8"))
    excluded = {key for q in eval_queries for key in q["relevant"]}

    pairs = make_pairs(docs, excluded)
    print(f"{len(pairs)} paires éligibles ({len(docs)} documents, {len(excluded)} exclus car pertinents "
          f"pour le set d'évaluation)")

    import random
    random.Random(args.seed).shuffle(pairs)
    sampled = pairs[:args.n]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for p in sampled:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(sampled)} paires écrites -> {OUT}")


if __name__ == "__main__":
    main()
