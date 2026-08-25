# FedMed v2.0 — Enterprise Deployment Guide

## Overview

FedMed v2.0 supports two enterprise deployment modes:
1. **Containerized Multi-Container Stack (Docker Compose)**
2. **Production Kubernetes Cluster (k8s / Helm)**

---

## Deployment Architectures

```
+-----------------------------------------------------------------------+
|                           KUBERNETES INGRESS                          |
+-----------------------------------------------------------------------+
                                   |
           +-----------------------+-----------------------+
           |                                               |
+-----------------------+                       +-----------------------+
|  FedMed Backend (x2)  |                       |  FedMed Flower Server |
|      (ClusterIP)      |                       |      (ClusterIP)      |
+-----------------------+                       +-----------------------+
           |                                               |
           +-----------------------+-----------------------+
                                   |
+-----------------------------------------------------------------------+
|                      MULTI-HOSPITAL EDGE RUNTIME                      |
| (Hospital Alpha, Hospital Beta, Hospital Gamma, Hospital Delta Nodes) |
+-----------------------------------------------------------------------+
```

---

## Authentication & RBAC

All administrative endpoints require JWT tokens:
- Role permissions: `Administrator`, `Hospital Administrator`, `Research Scientist`, `Auditor`, `Viewer`.
- Generate tokens via `POST /api/v1/auth/login`.
