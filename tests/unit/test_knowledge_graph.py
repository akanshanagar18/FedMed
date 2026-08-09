"""
Module: tests.unit.test_knowledge_graph

Purpose:
Unit test suite for SystemKnowledgeGraph (node/edge addition, lineage tracing, relationship querying).
"""

import pytest
from knowledge.graph import SystemKnowledgeGraph


def test_knowledge_graph_node_and_edge_addition():
    kg = SystemKnowledgeGraph()
    n1 = kg.add_node("hosp_gamma", "Hospital", "Hospital Silo Gamma")
    n2 = kg.add_node("model_v2", "Model", "BraTS MONAI UNet Candidate")
    e = kg.add_edge("hosp_gamma", "model_v2", "TRAINED")

    assert n1.id == "hosp_gamma"
    assert n2.id == "model_v2"
    assert e.source_id == "hosp_gamma"
    assert e.target_id == "model_v2"


def test_knowledge_graph_model_lineage_and_summary():
    kg = SystemKnowledgeGraph()
    summary = kg.get_summary()
    assert summary["total_nodes"] >= 5
    assert summary["total_edges"] >= 4
    assert "Hospital" in summary["node_type_breakdown"]

    lineage = kg.trace_model_lineage("model_prod")
    assert lineage["model_id"] == "model_prod"
    assert len(lineage["connected_hospitals"]) >= 2
