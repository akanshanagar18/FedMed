# API Contracts

**Parent Document:** [Platform Specification v1.1](platform_specification_v1.1.md)

All REST endpoints for FedMed are strictly versioned. Currently, all valid routes are prefixed with `/api/v1/`.

## Endpoints

| Endpoint | Method | Purpose | Payload Schema | Response Schema |
|---|---|---|---|---|
| `/health` | `GET` | System uptime and DB status | None | `NodeHealth` |
| `/nodes` | `GET` | List connected hospitals | None | `List[HospitalStatus]` |
| `/nodes/register` | `POST` | Register a new hospital | `HospitalStatus` | `SuccessResponse` |
| `/experiments` | `GET` | List all historical experiments | None | `List[Experiment]` |
| `/metrics` | `POST` | Ingest round metrics from FL | `TrainingMetric` | `SuccessResponse` |
| `/round/status` | `POST` | Update aggregation status | `TrainingRound` | `SuccessResponse` |
| `/dashboard` | `GET` | Initial hydration for UI | None | `DashboardSummary` |

## Standard Error Response
If any API fails, it will return a standardized error JSON:
```json
{
  "status_code": 422,
  "error_code": "VALIDATION_ERROR",
  "message": "The provided hyperparameter 'learning_rate' must be > 0.",
  "developer_details": "loc: ['body', 'learning_rate']",
  "recovery_suggestions": "Check experiment configuration."
}
```

> [!WARNING]
> Do NOT change these endpoints without submitting an RFC and receiving approval.
