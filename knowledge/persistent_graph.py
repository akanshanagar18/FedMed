"""
Module: knowledge.persistent_graph

Purpose:
Persistent Knowledge Graph for FedMed v2.0.
Provides SQLite-backed persistence (fedmed_kg.db) with a Neo4j-compatible GraphStoreDriver interface,
node & edge versioning, experiment/model/hospital/deployment lineage, and historical time-travel queries.
"""

import abc
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional


class GraphStoreDriver(abc.ABC):
    """Abstract Storage Driver Interface for Knowledge Graph Persistence (SQLite, Neo4j, etc.)."""

    @abc.abstractmethod
    def save_node(self, node_id: str, node_type: str, label: str, properties: Dict[str, Any], version: int, timestamp: float) -> None:
        pass

    @abc.abstractmethod
    def save_edge(self, source_id: str, target_id: str, relationship: str, properties: Dict[str, Any], version: int, timestamp: float) -> None:
        pass

    @abc.abstractmethod
    def get_node_as_of(self, node_id: str, timestamp: float) -> Optional[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def query_relationships_as_of(self, node_id: str, timestamp: float) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def get_all_nodes(self) -> List[Dict[str, Any]]:

        pass

    @abc.abstractmethod
    def get_all_edges(self) -> List[Dict[str, Any]]:
        pass


class SQLiteGraphDriver(GraphStoreDriver):
    """
    SQLite implementation of GraphStoreDriver managing 'fedmed_kg.db'.
    """

    def __init__(self, db_path: str = "fedmed_kg.db"):
        self.db_path = os.path.abspath(db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kg_nodes_versioned (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    properties_json TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    timestamp REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kg_edges_versioned (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship TEXT NOT NULL,
                    properties_json TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()

    def save_node(self, node_id: str, node_type: str, label: str, properties: Dict[str, Any], version: int, timestamp: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO kg_nodes_versioned (node_id, node_type, label, properties_json, version, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (node_id, node_type, label, json.dumps(properties), version, timestamp)
            )
            conn.commit()

    def save_edge(self, source_id: str, target_id: str, relationship: str, properties: Dict[str, Any], version: int, timestamp: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO kg_edges_versioned (source_id, target_id, relationship, properties_json, version, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (source_id, target_id, relationship, json.dumps(properties), version, timestamp)
            )
            conn.commit()

    def get_node_as_of(self, node_id: str, timestamp: float) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT node_id, node_type, label, properties_json, version, timestamp FROM kg_nodes_versioned WHERE node_id = ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
                (node_id, timestamp)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "node_id": row[0],
                    "node_type": row[1],
                    "label": row[2],
                    "properties": json.loads(row[3]),
                    "version": row[4],
                    "timestamp": row[5],
                }
        return None

    def query_relationships_as_of(self, node_id: str, timestamp: float) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT source_id, target_id, relationship, properties_json, version, timestamp FROM kg_edges_versioned WHERE (source_id = ? OR target_id = ?) AND timestamp <= ?",
                (node_id, node_id, timestamp)
            )
            rows = cursor.fetchall()
            edges = []
            for r in rows:
                edges.append({
                    "source_id": r[0],
                    "target_id": r[1],
                    "relationship": r[2],
                    "properties": json.loads(r[3]),
                    "version": r[4],
                    "timestamp": r[5],
                })
            return {"node_id": node_id, "as_of_timestamp": timestamp, "edges": edges}

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT node_id, node_type, label, properties_json, version, timestamp FROM kg_nodes_versioned GROUP BY node_id ORDER BY id DESC")
            rows = cursor.fetchall()
            return [{"node_id": r[0], "node_type": r[1], "label": r[2], "properties": json.loads(r[3]), "version": r[4], "timestamp": r[5]} for r in rows]

    def get_all_edges(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT source_id, target_id, relationship, properties_json, version, timestamp FROM kg_edges_versioned ORDER BY id DESC")
            rows = cursor.fetchall()
            return [{"source_id": r[0], "target_id": r[1], "relationship": r[2], "properties": json.loads(r[3]), "version": r[4], "timestamp": r[5]} for r in rows]


class PersistentKnowledgeGraph:
    """
    Persistent Knowledge Graph Manager supporting time-travel queries and versioning.
    """

    def __init__(self, driver: Optional[GraphStoreDriver] = None, db_path: Optional[str] = None):
        if driver is None:
            if db_path is not None:
                driver = SQLiteGraphDriver(db_path=db_path)
            else:
                driver = SQLiteGraphDriver()
        self.driver = driver
        self._seed_default_persistent_nodes()


    def _seed_default_persistent_nodes(self):
        ts = time.time()
        if not self.driver.get_node_as_of("hosp_alpha", ts):
            self.add_node("hosp_alpha", "Hospital", "Hospital Silo Alpha", {"scanner": "Siemens PRISMA 3T"})
            self.add_node("hosp_beta", "Hospital", "Hospital Silo Beta", {"scanner": "GE Discovery MR750 3T"})
            self.add_node("model_v2", "Model", "BraTS MONAI UNet v2.0", {"stage": "Production"})
            self.add_edge("hosp_alpha", "model_v2", "CONTRIBUTED_TO")

    def add_node(self, node_id: str, node_type: str, label: str, properties: Optional[Dict[str, Any]] = None, version: int = 1) -> None:
        self.driver.save_node(node_id, node_type, label, properties or {}, version, time.time())

    def add_edge(self, source_id: str, target_id: str, relationship: str, properties: Optional[Dict[str, Any]] = None, version: int = 1) -> None:
        self.driver.save_edge(source_id, target_id, relationship, properties or {}, version, time.time())

    def query_as_of(self, timestamp: float) -> Dict[str, Any]:
        """Time-travel query returning state of graph as of a historical timestamp."""
        nodes = [n for n in self.driver.get_all_nodes() if n.get("timestamp", 0) <= timestamp]
        edges = [e for e in self.driver.get_all_edges() if e.get("timestamp", 0) <= timestamp]
        return {
            "as_of_timestamp": timestamp,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    def get_lineage_summary(self) -> Dict[str, Any]:
        """Returns high level summary of stored knowledge graph lineage."""
        return {
            "total_nodes": len(self.driver.get_all_nodes()),
            "total_edges": len(self.driver.get_all_edges()),
            "lineage_active": True,
        }



global_persistent_graph = PersistentKnowledgeGraph()


