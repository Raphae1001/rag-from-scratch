import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _common import DEFAULT_EMBED_MODEL, add_embedder_args, build_embedder, make_parser  # noqa: E402
from rag.embed import HashingEmbedder  # noqa: E402


def _parse(argv):
    ap = make_parser("doc du script")
    add_embedder_args(ap)
    return ap.parse_args(argv)


def test_defaults_use_the_real_model():
    args = _parse([])
    assert args.fake is False
    assert args.model == DEFAULT_EMBED_MODEL


def test_fake_flag_builds_the_offline_embedder():
    assert isinstance(build_embedder(_parse(["--fake"])), HashingEmbedder)


def test_parser_keeps_script_docstring_as_help():
    assert make_parser("ligne 1\nligne 2").description == "ligne 1\nligne 2"
