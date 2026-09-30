"""Construit data/corpus.jsonl à partir des docs FastAPI + Starlette + Pydantic.

Chaque document = une section (## ou ###) d'une page Markdown.
Les titres '#' à l'intérieur des blocs de code sont ignorés.

Usage: python scripts/build_corpus.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "corpus.jsonl"
MIN_WORDS = 15

# name -> (dossier des .md, chemins exclus : fichiers exacts ou préfixes de dossier finissant par '/')
# Exclus : changelogs (URLs de PR = bruit), pages "méta" (people, newsletter, contribuer),
# et pydantic/api/ (stubs mkdocstrings `::: pydantic.X`, sans texte rédigé).
SOURCES = {
    "fastapi": (
        DATA / "fastapi_repo" / "docs" / "en" / "docs",
        {"release-notes.md", "_llm-test.md", "external-links.md", "fastapi-people.md",
         "management.md", "newsletter.md", "translations.md", "translation-banner.md",
         "contributing.md", "help-fastapi.md"},
    ),
    "starlette": (
        DATA / "starlette_repo" / "docs",
        {"release-notes.md", "third-party-packages.md", "contributing.md"},
    ),
    "pydantic": (
        DATA / "pydantic_repo" / "docs",
        {"api/", "pydantic_people.md", "contributing.md", "help_with_pydantic.md",
         "enterprise-support.md"},
    ),
}

HEADING = re.compile(r"^(#{1,3})\s+(.*?)\s*(\{\s*#[^}]*\})?\s*$")  # ancre `{ #id }` retirée

# FastAPI référence du code réel (exemples dans docs_src/, ou son propre code source ex. fastapi/openapi/)
# depuis ses pages Markdown via une directive de macro : `{* ../../<chemin> ln[a:b,c:d] hl[...] title[...] *}`
# (ln = lignes à inclure, les autres attributs — hl, title — sont ignorés ici, l'ordre et la présence de
# chacun varient). Sans résolution, ~23% des documents FastAPI du corpus (37,6% des documents FastAPI
# seuls) avaient une section avec du texte explicatif mais sans le code réel — mesuré après une génération
# hallucinée à cause d'un contexte incomplet (voir README, section Phase 4).
# On résout <chemin> depuis la racine du dépôt plutôt que de recompter les "../" (dont la profondeur
# relative au fichier .md ne correspond pas à la macro réelle de FastAPI) : qu'il s'agisse de docs_src/
# (exemples) ou de fastapi/openapi/ (le framework lui-même), le chemin après le(s) "../" est toujours
# relatif à la racine du dépôt. Les attributs sont capturés en bloc (`attrs`) puis on y cherche `ln[...]`
# séparément, plutôt que de figer leur ordre/présence dans le regex principal (une combinaison — ex.
# `hl[...] title[...]` sans `ln[]` — a fait échouer une première version qui supposait `ln` avant `hl`).
CODE_SNIPPET = re.compile(r"\{\*\s*(?:\.\./)+(\S+?)((?:\s+\w+\[[^\]]*\])*)\s*\*\}")
LN_ATTR = re.compile(r"\bln\[([^\]]+)\]")


def _parse_line_ranges(spec: str) -> list[tuple[int, int]]:
    """"1:9,29:35" ou "16" -> [(1, 9), (29, 35)] ou [(16, 16)] (1-indexé, inclusif des deux bornes)."""
    ranges = []
    for part in spec.split(","):
        part = part.strip()
        a, _, b = part.partition(":")
        ranges.append((int(a), int(b) if b else int(a)))
    return ranges


def resolve_code_snippets(text: str, repo_root: Path) -> str:
    """Remplace chaque directive `{* <chemin depuis la racine du dépôt>.py ... *}` par le code réel."""
    def repl(m):
        rel_path, attrs = m.group(1), m.group(2)
        ln_match = LN_ATTR.search(attrs)
        ln_spec = ln_match.group(1) if ln_match else None
        target = repo_root / rel_path
        if not target.exists():
            return m.group(0)  # fichier introuvable (sparse-checkout incomplet) : laisser visible, pas silencieux
        lines = target.read_text(encoding="utf-8").splitlines()
        if ln_spec:
            selected = [ln for a, b in _parse_line_ranges(ln_spec) for ln in lines[a - 1:b]]
        else:
            selected = lines
        lang = target.suffix.lstrip(".") or "text"
        return "\n```{}\n{}\n```\n".format(lang, "\n".join(selected))
    return CODE_SNIPPET.sub(repl, text)


def is_excluded(rel: str, excluded: set[str]) -> bool:
    return any(rel == e or (e.endswith("/") and rel.startswith(e)) for e in excluded)


def split_page(text: str):
    """Yield (section_title, body) pour chaque section ##/### d'une page."""
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)  # front matter
    section, buf, in_code = "", [], False

    def flush():
        body = "\n".join(buf).strip()
        return (section, body) if body else None

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if m:
            level, heading = len(m.group(1)), m.group(2)
            if level == 1:
                continue
            out = flush()
            if out:
                yield out
            section, buf = heading, []
        else:
            buf.append(line)
    out = flush()
    if out:
        yield out


def main():
    docs, skipped = [], 0
    per_source = {}
    for source, (root, excluded) in SOURCES.items():
        if not root.exists():
            raise SystemExit(f"Dossier manquant : {root} (voir README, section Démarrage)")
        for path in sorted(root.rglob("*.md")):
            rel = path.relative_to(root).as_posix()
            if is_excluded(rel, excluded):
                continue
            text = path.read_text(encoding="utf-8")
            if source == "fastapi":
                text = resolve_code_snippets(text, DATA / "fastapi_repo")
            page_title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), path.stem)
            for section, body in split_page(text):
                if len(body.split()) < MIN_WORDS:
                    skipped += 1
                    continue
                docs.append({
                    "id": len(docs),
                    "source": source,
                    "path": f"{source}/{rel}",
                    "page": page_title,
                    "section": section or "(introduction)",
                    # titre de page + section en tête : donne du contexte aux sections sans titre
                    "text": "\n".join(part for part in (page_title, section, body) if part),
                })
                per_source[source] = per_source.get(source, 0) + 1
    with OUT.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(docs)} documents -> {OUT} ({skipped} sections < {MIN_WORDS} mots ignorées)")
    print("par source :", per_source)


if __name__ == "__main__":
    main()
