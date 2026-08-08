# FedMed v2.0 — Kubernetes Deployment & Topology Guide

## Overview
This document details raw Kubernetes manifest deployment topology, PersistentVolumeClaims, Ingress rules, NetworkPolicies, and Autoscaling for **FedMed v2.0**.

---

## 1. Kubernetes Architecture Diagram

```mermaid
graph TD
    Ingress[Ingress Controller (fedmed.internal)] --> DashboardSvc[Service: fedmed-dashboard]
    Ingress --> BackendSvc[Service: fedmed-backend]
    
    DashboardSvc --> DashboardPod[Pods: fedmed-dashboard]
    BackendSvc --> BackendPod[Pods: fedmed-backend]
    
    HospitalAlpha[Pod: hospital-alpha] --> FlowerSvc[Service: fedmed-flower]
    HospitalBeta[Pod: hospital-beta] --> FlowerSvc
    HospitalGamma[Pod: hospital-gamma] --> FlowerSvc
    
    FlowerSvc --> FlowerPod[Pod: fedmed-flower-server]

    BackendPod --> DB_PVC[(PVC: fedmed-database-pvc)]
    BackendPod --> Checkpoints_PVC[(PVC: fedmed-checkpoints-pvc)]
    FlowerPod --> Checkpoints_PVC
```

---

## 2. Deploying via Raw Kubernetes Manifests (`kubectl`)

### Step 1: Create Namespace & Secrets
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
```

### Step 2: Provision Persistent Storage
```bash
kubectl apply -f k8s/pvc.yaml
```

### Step 3: Deploy Microservices & Services
```bash
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/flower-server-deployment.yaml
kubectl apply -f k8s/dashboard-deployment.yaml
kubectl apply -f k8s/hospitals-deployment.yaml
kubectl apply -f k8s/services.yaml
```

### Step 4: Apply Networking & HPA
```bash
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/network-policy.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/disaster-recovery-backup.yaml
```

---

## 3. Verification Commands
```bash
# Check pod deployment status
kubectl get pods -n fedmed

# Check HPA autoscaling
kubectl get hpa -n fedmed

# View backend logs
kubectl logs -n fedmed deployment/fedmed-backend -f
```
