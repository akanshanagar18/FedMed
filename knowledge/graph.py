"""
Module: knowledge.graph

Purpose:
Internal Property System Knowledge Graph connecting platform entities across the FL lifecycle.
Exposes APIs for lineage tracing, incident graph traversals, and impact radius estimation.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Set


@dataclass
class GraphNode:
    id: str
    node_type: str  # Hospital, Round, Experiment, Model, Metric, DriftEvent, GovernanceDecision, PrivacyEvent, Deployment, Certificate
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "label": self.label,
            "properties": self.properties,
            "created_at": self.created_at,
        }


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relationship: str  # PARTICIPATES_IN, PRODUCES, TRIGGERS, GOVERNS, DEPLOYED_VIA, CERTIFIES
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
            "properties": self.properties,
            "created_at": self.created_at,
        }


class SystemKnowledgeGraph:
    """
    In-memory Property Knowledge Graph storing entities and relationships across FedMed.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SystemKnowledgeGraph, cls).__new__(cls)
            cls._instance.nodes: Dict[str, GraphNode] = {}
            cls._instance.edges: List[GraphEdge] = []
            cls._instance._seed_default_graph()
        return cls._instance

    def _seed_default_graph(self):
        """Seeds initial default node relationships."""
        self.add_node("hosp_alpha", "Hospital", "Hospital Silo Alpha", {"scanner": "Siemens PRISMA 3T"})
        self.add_node("hosp_beta", "Hospital", "Hospital Silo Beta", {"scanner": "GE Discovery MR750 3T"})
        self.add_node("exp_brats_v2", "Experiment", "BraTS 3D UNet Benchmark", {"status": "ACTIVE"})
        self.add_node("model_prod", "Model", "BraTS MONAI UNet v2.0", {"stage": "Production"})
        self.add_node("cert_hipaa", "Certificate", "HIPAA/GDPR Compliance Cert", {"hash": "abc123sha256"})

        self.add_edge("hosp_alpha", "exp_brats_v2", "PARTICIPATES_IN")
        self.add_edge("hosp_beta", "exp_brats_v2", "PARTICIPATES_IN")
        self.add_edge("exp_brats_v2", "model_prod", "PRODUCES")
        self.add_edge("model_prod", "cert_hipaa", "CERTIFIED_BY")

    def add_node(self, node_id: str, node_type: str, label: str, properties: Optional[Dict[str, Any]] = None) -> GraphNode:
        """Adds or updates a node in the Knowledge Graph."""
        node = GraphNode(
            id=node_id,
            node_type=node_type,
            label=label,
            properties=properties or {},
        )
        self.nodes[node_id] = node
        return node

    def add_edge(self, source_id: str, target_id: str, relationship: str, properties: Optional[Dict[str, Any]] = None) -> GraphEdge:
        """Creates a relationship edge between source and target nodes."""
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            properties=properties or {},
        )
        self.edges.append(edge)
        return edge

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single node by ID."""
        node = self.nodes.get(node_id)
        return node.to_dict() if node else None

    def query_relationships(self, node_id: str) -> Dict[str, Any]:
        """Returns all connected incoming and outgoing edges for a given node."""
        outgoing = [e.to_dict() for e in self.edges if e.source_id == node_id]
        incoming = [e.to_dict() for e in self.edges if e.target_id == node_id]
        return {
            "node_id": node_id,
            "outgoing_edges": outgoing,
            "incoming_edges": incoming,
        }

    def trace_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Traces complete lineage of a model back to its source hospitals and benchmark experiment."""
        model_node = self.nodes.get(model_id)
        connected_edges = [e for e in self.edges if e.target_id == model_id or e.source_id == model_id]

        hospitals = []
        for e in self.edges:
            if e.relationship == "PARTICIPATES_IN":
                h_node = self.nodes.get(e.source_id)
                if h_node and h_node.node_type == "Hospital":
                    hospitals.append(h_node.to_dict())

        return {
            "model_id": model_id,
            "model_details": model_node.to_dict() if model_node else None,
            "connected_hospitals": hospitals,
            "total_lineage_edges": len(connected_edges),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns high-level graph metrics and total counts."""
        node_counts = {}
        for n in self.nodes.values():
            node_counts[n.node_type] = node_counts.get(n.node_type, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_type_breakdown": node_counts,
            "sample_nodes": [n.to_dict() for n in list(self.nodes.values())[:10]],
            "sample_edges": [e.to_dict() for e in self.edges[:10]],
        }
