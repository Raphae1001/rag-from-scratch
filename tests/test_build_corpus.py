"""Résolution des directives {* docs_src/... *} de FastAPI en code réel (scripts/build_corpus.py).

~23% des documents FastAPI du corpus référencent un exemple de code via cette directive plutôt que de
l'inclure en Markdown ; sans résolution, ces sections perdent leur exemple concret (observé en Phase 4 :
une réponse a été refusée alors que le passage contenait une explication correcte mais sans le code)."""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_corpus import _parse_line_ranges, resolve_code_snippets  # noqa: E402


def test_parse_line_ranges_single_range():
    assert _parse_line_ranges("29:35") == [(29, 35)]


def test_parse_line_ranges_single_line_no_colon():
    assert _parse_line_ranges("16") == [(16, 16)]


def test_parse_line_ranges_multiple_comma_separated():
    assert _parse_line_ranges("1:9,29:35,42") == [(1, 9), (29, 35), (42, 42)]


def test_resolve_code_snippets_inserts_code_with_line_range(tmp_path):
    (tmp_path / "docs_src" / "tutorial").mkdir(parents=True)
    (tmp_path / "docs_src" / "tutorial" / "example.py").write_text(
        "\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")

    text = "Some prose.\n{* ../../docs_src/tutorial/example.py ln[2:4] *}\nMore prose."
    out = resolve_code_snippets(text, tmp_path)

    assert "```py\nline2\nline3\nline4\n```" in out
    assert "line1" not in out and "line5" not in out
    assert "Some prose." in out and "More prose." in out


def test_resolve_code_snippets_without_ln_includes_whole_file(tmp_path):
    (tmp_path / "docs_src").mkdir()
    (tmp_path / "docs_src" / "whole.py").write_text("a\nb\nc", encoding="utf-8")

    out = resolve_code_snippets("{* ../../docs_src/whole.py *}", tmp_path)
    assert "```py\na\nb\nc\n```" in out


def test_resolve_code_snippets_ignores_hl_highlight_spec(tmp_path):
    (tmp_path / "docs_src").mkdir()
    (tmp_path / "docs_src" / "f.py").write_text("x\ny\nz", encoding="utf-8")

    out = resolve_code_snippets("{* ../../docs_src/f.py ln[1:2] hl[1] *}", tmp_path)
    assert "```py\nx\ny\n```" in out


def test_resolve_code_snippets_leaves_directive_untouched_when_file_missing(tmp_path):
    text = "{* ../../docs_src/does/not/exist.py ln[1:2] *}"
    assert resolve_code_snippets(text, tmp_path) == text


def test_resolve_code_snippets_handles_hl_and_title_without_ln(tmp_path):
    """Variante observée dans bigger-applications.md : hl[...] et title[...], sans ln[] du tout, et pas
    dans le même ordre que ln/hl — le regex ne doit pas supposer un ordre ou une présence fixe."""
    (tmp_path / "docs_src").mkdir()
    (tmp_path / "docs_src" / "users.py").write_text("u1\nu2\nu3", encoding="utf-8")

    out = resolve_code_snippets('{* ../../docs_src/users.py hl[1,3] title["app/routers/users.py"] *}', tmp_path)
    assert "```py\nu1\nu2\nu3\n```" in out  # pas de ln[] -> fichier entier


def test_resolve_code_snippets_handles_multiple_directives_in_one_page(tmp_path):
    (tmp_path / "docs_src").mkdir()
    (tmp_path / "docs_src" / "a.py").write_text("aaa", encoding="utf-8")
    (tmp_path / "docs_src" / "b.py").write_text("bbb", encoding="utf-8")

    text = "{* ../../docs_src/a.py *}\n{* ../../docs_src/b.py *}"
    out = resolve_code_snippets(text, tmp_path)
    assert "aaa" in out and "bbb" in out


def test_resolve_code_snippets_resolves_paths_outside_docs_src_too(tmp_path):
    """Un cas réel du corpus (configure-swagger-ui.md) référence le code source de FastAPI lui-même,
    pas un exemple sous docs_src/ : {* ../../fastapi/openapi/docs.py ln[9:24] hl[18:24] *}. Le résolveur
    ne doit pas être limité à "docs_src/" -- tout chemin relatif à la racine du dépôt doit marcher."""
    (tmp_path / "fastapi" / "openapi").mkdir(parents=True)
    (tmp_path / "fastapi" / "openapi" / "docs.py").write_text(
        "\n".join(f"line{i}" for i in range(1, 30)), encoding="utf-8")

    out = resolve_code_snippets("{* ../../fastapi/openapi/docs.py ln[9:11] hl[10] *}", tmp_path)
    assert "```py\nline9\nline10\nline11\n```" in out
