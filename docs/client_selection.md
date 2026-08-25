# Client Selection Engine

## Overview
This document details the client selection policies implemented in **FedMed v2.0** (`server/client_selection.py`).

---

## 1. Selection Policies

- **RandomSelector:** Uniform random sampling across all active hospital nodes.
- **ResourceAwareSelector:** Selects nodes based on CPU cores, RAM GB, and GPU availability scores.
- **DataAwareSelector:** Selects nodes with the highest patient dataset sample volumes.
- **FairScheduler:** Min-participation scheduling to prevent straggler starvation.
- **ReputationBasedSelector:** Selects nodes based on high uptime ratio and low historical latency/failures.

---

## 2. Usage Example

```python
from server.client_selection import ClientSelectionEngine, ClientProfile

selector = ClientSelectionEngine.get_selector("resource_aware")
selected = selector.select_clients(available_clients, num_to_select=10)
```
