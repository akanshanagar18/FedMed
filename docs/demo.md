# 3-Minute Live Demonstration Script — FedMed v1.0

This script provides a 3-minute demonstration walkthrough for judges, reviewers, and evaluators.

---

## ⏱️ Minute 0:00 - 0:45 | Problem Statement & Solution

**What to Say:**
> "Welcome! Today we present **FedMed**, a cross-silo Federated Learning platform engineered for medical image segmentation. Healthcare data is siloed across hospitals due to privacy laws like HIPAA. FedMed allows multiple hospitals to collaboratively train a 3D U-Net model on private MRI scans without ever sharing raw patient data."

**What to Show:**
* Open [README.md](file:///Users/siddhant_patil/Projects/FedMed/README.md) architecture diagram or system overview. Point out **Hospital Silos**, **Flower Central Aggregator**, **TenSEAL Encryption**, and the **Grafana Dashboard**.

---

## ⏱️ Minute 0:45 - 1:30 | Single-Command Execution

**What to Say:**
> "Watch how simple it is to deploy the complete platform. With a single command, our production orchestrator checks for port collisions, starts our FastAPI backend, initializes database tables, launches our Flower central aggregator, and connects Hospital Node Alpha and Hospital Node Beta."

**What to Do:**
1. Open terminal and run:
   ```bash
   python3 scripts/run_simulation.py
   ```
2. Point out structured logging tags: `[ORCHESTRATOR]`, `[BACKEND]`, `[FLOWER]`, `[CLIENT-A]`, `[CLIENT-B]`.
3. Highlight active readiness polling: `FastAPI Backend active & healthy!`.

---

## ⏱️ Minute 1:30 - 2:30 | Real-time Telemetry Dashboard

**What to Say:**
> "Now let's open our live Grafana-style telemetry dashboard at port 8000. As training progresses across federated rounds, watch our Recharts loss curve drop in real time via WebSockets, while the Dice similarity coefficient increases. We also see live latency indicators for Hospital Alpha and Hospital Beta."

**What to Show:**
1. Open `http://127.0.0.1:8000/` in browser.
2. Show the top status bar: `REST API 200 OK`, `DATABASE SQLite Active`, `WEBSOCKET LIVE`, `FLOWER 127.0.0.1:8080`.
3. Show the live Recharts graph updating as Round 1, Round 2, and Round 3 complete.
4. Point out the Hospital Client Silos table showing active connection states and latency metrics.

---

## ⏱️ Minute 2:30 - 3:00 | Graceful Shutdown & Wrap-Up

**What to Say:**
> "Notice how upon completion of the 3 federated rounds, the orchestrator automatically records metrics in SQLite and executes a clean process tree shutdown with zero orphan processes remaining. Thank you, and we welcome your questions!"

**What to Show:**
* Show final terminal output: `SUCCESS: Federated Learning Simulation completed 3 rounds cleanly! All managed subprocesses terminated cleanly.`
