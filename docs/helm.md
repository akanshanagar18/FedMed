# FedMed v2.0 — Helm v3 Chart Guide

## Overview
This document describes how to package, template, install, upgrade, and rollback **FedMed v2.0** using Helm v3.

---

## 1. Helm Chart Overview (`helm/fedmed/`)
- `Chart.yaml`: Helm v3 metadata (version 2.0.0).
- `values.yaml`: Configurable defaults (replicas, CPU/Memory limits, storage classes, hospital client nodes).
- `templates/`: Dynamic Go templates for deployments, services, ingress, and configmaps.

---

## 2. Helm Installation Workflow

### Linting Chart
```bash
helm lint helm/fedmed/
```

### Dry-Run Template Rendering
```bash
helm template release-test helm/fedmed/
```

### Deploying Release
```bash
helm install fedmed-release helm/fedmed/ --namespace fedmed --create-namespace
```

### Upgrading Release
```bash
helm upgrade fedmed-release helm/fedmed/ --set backend.replicaCount=3
```

### Rollback Release
```bash
helm rollback fedmed-release 1
```
