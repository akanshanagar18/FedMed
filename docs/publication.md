# FedMed v2.0 — Publication & Research Benchmarking Architecture

## Overview
This document describes the publication benchmark suite pipeline, LaTeX export format, and research paper reproducibility integration.

## Workflow Sequence
```mermaid
sequenceDiagram
    autonumber
    actor Scientist as Research Scientist
    participant Script as Benchmark Orchestrator (run_benchmarks.py)
    participant Stat as Statistical Analysis Engine
    participant Vis as Benchmark Visualizer (Matplotlib/Seaborn)
    participant Export as Publication Export Engine
    participant REST as FastAPI REST API

    Scientist->>Script: Launch Sweep (strategies × partitions × privacy × seeds)
    Script->>Stat: Compute Descriptive Stats & Inferential Tests (p-values, Cohen's d)
    Script->>Vis: Generate 10 Publication Figures (PNG & PDF)
    Script->>Export: Build PDF, Markdown, LaTeX (tables.tex), and CSV Artifacts
    Script->>REST: Register Benchmark & Analytics Results
```

## LaTeX Table Integration (`tables.tex`)
Include `tables.tex` directly in your LaTeX research manuscript:

```latex
\input{results/benchmark_bm_1786208050/tables.tex}
```

## REST API Analytics Endpoints
```http
GET /api/v1/benchmarks/{id}/analytics
GET /api/v1/benchmarks/{id}/statistical-tests
GET /api/v1/benchmarks/{id}/latex-tables
GET /api/v1/benchmarks/{id}/plots
```
