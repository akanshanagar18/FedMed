# Network Profile Simulator

## Overview
This document details the network bottleneck and latency simulator implemented in **FedMed v2.0** (`simulation/network_simulator.py`).

---

## 1. Network Profile Presets (`configs/network/`)

| Profile | Latency (ms) | Bandwidth (Mbps) | Packet Loss (%) | Drop Prob |
|---|---|---|---|---|
| **LAN** | 0.5 ms | 1000.0 Mbps | 0.0% | 0.00 |
| **WiFi** | 15.0 ms | 100.0 Mbps | 0.1% | 0.01 |
| **Mobile 4G** | 50.0 ms | 20.0 Mbps | 1.0% | 0.05 |
| **Mobile 5G** | 10.0 ms | 200.0 Mbps | 0.1% | 0.005 |
| **Satellite** | 600.0 ms | 10.0 Mbps | 3.0% | 0.10 |
| **Poor Network**| 200.0 ms | 2.0 Mbps | 5.0% | 0.15 |

---

## 2. Configuration & Execution

```bash
python scripts/run_simulation.py --network mobile_4g
python scripts/run_simulation.py --network satellite
```
