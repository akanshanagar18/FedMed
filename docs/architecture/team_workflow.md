# Team Workflow & Governance

**Parent Document:** [Platform Specification v1.1](platform_specification_v1.1.md)

## Governance Rules
- **Branch Strategy:** Gitflow. `main` is production. `development` is integration. 
- **Collaboration:** Features use `feature/[module]-[description]` (e.g., `feature/model-unet3d`).
- **Pull Requests:** Require 1 approval from a different module owner. Code must pass `black`, `mypy`, and `pytest`.
- **Contract Evolution:** If an API contract (in `common/contracts`) needs to change, it **requires a meeting** with Siddhant. Backend schemas are immutable without consensus.

## Definition of Done (DoD)
Before a PR is merged into `development`, the following must be true:
1. Code is fully implemented.
2. Unit tests are written and passing.
3. Docstrings are added for all public functions/classes.
4. Type hints pass `mypy` without errors.
5. No circular imports exist.
6. The PR has been reviewed and approved.
