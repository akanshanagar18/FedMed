# FedMed v2.0 — Kubernetes & Helm Guide

## Production Kubernetes Manifests

All Kubernetes deployment files are stored in `k8s/`:

```bash
# Create namespace
kubectl apply -f k8s/fedmed-namespace.yaml

# Deploy all services, deployments, ingress, and horizontal autoscalers
kubectl apply -f k8s/
```

### Scaling & Autoscaling (HPA)
The `fedmed-backend-hpa.yaml` automatically scales the backend API deployment between 2 and 10 replicas when CPU usage exceeds 80%:

```yaml
spec:
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
```
