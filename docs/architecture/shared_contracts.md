# Shared Contracts Reference

**Parent Document:** [Platform Specification v1.1](platform_specification_v1.1.md)

## What is `common/contracts/`?
To eliminate tight coupling across our four modules, we maintain a strictly versioned, collectively-owned module called `common/contracts/`.

## Core Schemas

- **`TrainingMetric`**: `experiment_id`, `round_number`, `training_loss`, `validation_loss`, `dice_score`, `iou`.
- **`HospitalStatus`**: `hospital_id`, `name`, `connection_status` (Enum), `client_latency_ms`.
- **`TrainingRound`**: `round_number`, `status` (Enum), `participating_hospitals`.
- **`NodeHealth`**: `status`, `active_connections`, `uptime_seconds`.
- **`Experiment`**: `experiment_id`, `name`, `hyperparameters`, `encryption_status`.

## Project Rules
- **Ownership:** Collective. No single person owns these files.
- **Modification:** You may ONLY add `Optional` fields without review.
- **Breaking Changes:** Deleting or renaming a field requires an RFC and approval from both Siddhant and Vishnu.
- **Dependencies:** This folder MUST NOT import from any other folder in the repository. It is the absolute bottom of the dependency graph.
