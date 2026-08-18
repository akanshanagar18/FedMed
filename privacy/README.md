# Privacy Module

## Components

- communication.py
- aggregation.py
- benchmark.py

## Purpose

This module prepares the secure aggregation layer for the FedMed federated learning workflow.

## Planned Workflow

Encrypted client updates
        ↓
receive_encrypted_updates()
        ↓
aggregate_encrypted_updates()
        ↓
send_global_model()