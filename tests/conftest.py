import pytest
import duckdb
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from energy_cross_commodity.db import init_db, seed_commodities, get_connection


@pytest.fixture
def db_conn():
    conn = duckdb.connect(":memory:")
    init_db(conn)
    seed_commodities(conn)
    yield conn
    conn.close()
