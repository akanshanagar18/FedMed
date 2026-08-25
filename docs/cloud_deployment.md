# FedMed v2.0 — Multi-Cloud Deployment Guide (GKE, AKS, EKS)

## Overview
This document covers cloud-native deployment patterns for **Google Kubernetes Engine (GKE)**, **Azure Kubernetes Service (AKS)**, and **Amazon EKS**.

---

## 1. GKE Deployment (Google Cloud)
```bash
# Authenticate GKE Cluster
gcloud container clusters get-credentials fedmed-cluster --region us-central1 --project fedmed-gcp

# Apply Manifests
kubectl apply -f k8s/
```

## 2. EKS Deployment (AWS)
```bash
aws eks update-kubeconfig --region us-west-2 --name fedmed-eks
helm install fedmed helm/fedmed/
```

## 3. Disaster Recovery & Restoration Workflow
In the event of volume corruption:
1. Locate latest backup snapshot in `/app/results/backups/`.
2. Restore SQLite database: `cp /app/results/backups/<TS>/fedmed.db /app/data_db/`.
3. Restart backend deployment: `kubectl rollout restart deployment/fedmed-backend -n fedmed`.
