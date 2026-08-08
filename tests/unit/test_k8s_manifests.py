"""
Module: tests.unit.test_k8s_manifests

Purpose:
Unit test suite validating Kubernetes manifests syntax, apiVersions, metadata, and container configurations under k8s/.
"""

import os
import glob
import yaml
import pytest


def test_k8s_manifest_files_exist_and_parse():
    manifest_files = glob.glob("k8s/*.yaml")
    assert len(manifest_files) >= 10

    for path in manifest_files:
        with open(path, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
            assert len(docs) >= 1
            for doc in docs:
                if doc is None:
                    continue
                assert "apiVersion" in doc
                assert "kind" in doc
                assert "metadata" in doc
                assert "name" in doc["metadata"]


def test_k8s_deployments_have_probes_and_resources():
    deployments = ["k8s/backend-deployment.yaml", "k8s/flower-server-deployment.yaml", "k8s/dashboard-deployment.yaml"]
    for path in deployments:
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
            containers = doc["spec"]["template"]["spec"]["containers"]
            assert len(containers) >= 1
            container = containers[0]

            assert "readinessProbe" in container or "livenessProbe" in container
            assert "resources" in container
            assert "requests" in container["resources"]
            assert "limits" in container["resources"]
