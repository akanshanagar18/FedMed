# Distributed Systems Runtime

## Overview
This document describes the large-scale distributed federated learning runtime architecture in **FedMed v2.0**, scaling seamlessly from 3 to 100 hospital client nodes across heterogeneous network environments.

---

## Architecture Topology

```
+-----------------------------------------------------------------------------------+
|                            FedMed Central Flower Server                           |
|      (Client Selection Engine + Network Simulator + Asynchronous Aggregator)      |
+-----------------------------------------------------------------------------------+
       ^                                    ^                                    ^
       | (Quantized / Sparsified)           | (SignSGD + EF)                     | (Plaintext / DP / HE)
       v                                    v                                    v
+------------------+                 +------------------+                 +------------------+
| Hospital Node 1  |                 | Hospital Node 2  |                 | Hospital Node N  |
| (Mobile 4G)      |                 | (WiFi)           |                 | (Satellite)      |
+------------------+                 +------------------+                 +------------------+
```
