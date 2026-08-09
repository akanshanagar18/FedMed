"""
Module: tests.unit.test_persistent_graph

Purpose:
Unit test suite for PersistentKnowledgeGraph (SQLite persistence, node/edge versioning, time-travel queries).
"""

import time
import pytest
from knowledge.persistent_graph import PersistentKnowledgeGraph


def test_persistent_knowledge_graph_operations():
    kg = PersistentKnowledgeGraph()
    t_start = time.time()

    kg.add_node("node_test_1", "Model", "Test Model", {"accuracy": 0.85})
    kg.add_node("node_test_2", "Hospital", "Test Hospital", {})
    kg.add_edge("node_test_2", "node_test_1", "TRAINED")

    query_now = kg.query_as_of(time.time())
    assert query_now["total_nodes"] >= 2
    assert query_now["total_edges"] >= 1

    query_past = kg.query_as_of(t_start - 100)
    assert query_past["total_nodes"] == 0
