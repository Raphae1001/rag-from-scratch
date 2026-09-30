"""Fixtures DB partagées par test_db.py et test_api.py. Deux sources possibles, dans cet ordre :
  1. `pgserver` (paquet pip, Postgres embarqué et jetable) s'il est installé ;
  2. sinon le Postgres du `docker compose` (DATABASE_URL), dans une base DÉDIÉE `rag_test_pytest`
     créée puis supprimée : la base principale (et ses embeddings) n'est jamais touchée.
Sans aucun des deux, les tests qui en dépendent sont ignorés. RAG_TEST_USE_DATABASE_URL=1 force la source n°2."""
import os

import pytest

try:
    import pgserver
except ImportError:  # optionnel : sans lui, on utilise le Postgres du docker compose
    pgserver = None
pytest.importorskip("psycopg")
pytest.importorskip("pgvector")

import psycopg  # noqa: E402
from psycopg.conninfo import conninfo_to_dict, make_conninfo  # noqa: E402

from rag import db  # noqa: E402

TEST_DB = "rag_test_pytest"
SAFE_DB_NAMES = {TEST_DB, "postgres"}   # bases que les tests ont le droit de vider (postgres = pgserver jetable)


def _reachable_dsn():
    """DSN du Postgres réel (docker compose) ; None s'il est injoignable."""
    dsn = os.environ.get("DATABASE_URL", db.DEFAULT_DSN)
    try:
        with psycopg.connect(dsn, connect_timeout=3, autocommit=True) as c:
            c.execute("SELECT 1")
    except psycopg.Error:
        return None
    return dsn


@pytest.fixture(scope="module")
def dsn(tmp_path_factory):
    if pgserver is not None and os.environ.get("RAG_TEST_USE_DATABASE_URL") != "1":
        srv = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="stop")
        yield srv.get_uri()
        srv.cleanup()
        return
    server_dsn = _reachable_dsn()
    if server_dsn is None:
        pytest.skip("ni pgserver ni Postgres joignable (lancer `docker compose up -d`)")
    if conninfo_to_dict(server_dsn).get("dbname") == TEST_DB:
        pytest.skip(f"DATABASE_URL pointe déjà sur {TEST_DB} : refus de le supprimer")
    with psycopg.connect(server_dsn, autocommit=True) as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
        admin.execute(f'CREATE DATABASE "{TEST_DB}"')
    try:
        yield make_conninfo(server_dsn, dbname=TEST_DB)
    finally:
        with psycopg.connect(server_dsn, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')


@pytest.fixture()
def conn(dsn):
    c = db.connect(dsn)
    name = c.execute("SELECT current_database()").fetchone()[0]
    assert name in SAFE_DB_NAMES, f"garde-fou : refus de vider la base {name!r}"   # ne jamais toucher à `rag`
    db.init_schema(c)
    db.reset_data(c)
    db.drop_hnsw(c)
    yield c
    c.close()
