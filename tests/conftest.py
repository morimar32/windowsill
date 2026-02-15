import sys
import os

import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import database
from meta_rel import MetaRelScorer


@pytest.fixture(scope="session")
def db_con():
    """Read-only connection to the real database."""
    con = database.get_connection(config.DB_PATH, read_only=True)
    yield con
    con.close()


@pytest.fixture(scope="session")
def scorer(db_con):
    """Pre-loaded MetaRelScorer instance."""
    return MetaRelScorer(db_con)


@pytest.fixture(scope="session")
def edge_weights(db_con):
    """All reef_edges rows as dict[(src, tgt)] → {containment, lift, pos_similarity,
    valence_gap, specificity_gap, weight}."""
    rows = db_con.execute("""
        SELECT source_reef_id, target_reef_id,
               containment, lift, pos_similarity,
               valence_gap, specificity_gap, weight
        FROM reef_edges
    """).fetchall()
    edges = {}
    for r in rows:
        edges[(r[0], r[1])] = {
            "containment": r[2],
            "lift": r[3],
            "pos_similarity": r[4],
            "valence_gap": r[5],
            "specificity_gap": r[6],
            "weight": r[7],
        }
    return edges
