#!/usr/bin/env bash
# Récupère les docs FastAPI, Starlette et Pydantic à des commits FIGÉS (corpus reproductible).
# Usage: bash scripts/fetch_docs.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

fetch() {  # nom  url  sha  dossier(s) à checkout (un ou plusieurs, ex: "docs/en/docs" "docs_src")
  local name=$1 url=$2 sha=$3 dir="data/${1}_repo"
  shift 3
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" = "$sha" ]; then
    git -C "$dir" sparse-checkout set "$@"   # idempotent : rafraîchit si la liste de dossiers a changé
    echo "$name: déjà au commit $sha"; return
  fi
  rm -rf "$dir"
  git clone -q --filter=blob:none --no-checkout "$url" "$dir"
  git -C "$dir" sparse-checkout set "$@"
  git -C "$dir" fetch -q --depth 1 origin "$sha"
  git -C "$dir" checkout -q "$sha"
  echo "$name: $sha"
}

# fastapi : docs_src/ (exemples de code) et fastapi/openapi/ (une directive référence le code source du
# framework lui-même, pas un exemple) en plus de docs/en/docs — voir build_corpus.py::resolve_code_snippets.
fetch fastapi   https://github.com/fastapi/fastapi.git   a3d205bf19640528718cb4f05ab77f4dfca6ad9a docs/en/docs docs_src fastapi/openapi
fetch starlette https://github.com/encode/starlette.git  63c5760d8a672cee96e1e523d84bfa1c77d9ee4c docs
fetch pydantic  https://github.com/pydantic/pydantic.git bb6da4cfbb1f559885ea2fa207ec93853bfeac64 docs
