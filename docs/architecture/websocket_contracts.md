# WebSocket Contracts

**Parent Document:** [Platform Specification v1.1](platform_specification_v1.1.md)

The central monitoring WebSocket endpoint is located at `ws://<host>/ws/telemetry`.
This is a multiplexed stream. Clients listen to specific event names.

## Events

| Event Name | Producer | Consumer | Payload Schema | Purpose |
|---|---|---|---|---|
| `training_started` | FL Server | Dashboard | `TrainingRound` | Notifies round begin. |
| `metrics_updated` | FL Server | Dashboard | `TrainingMetric`| Streams live loss/dice. |
| `hospital_connected` | FL Client | Dashboard | `HospitalStatus`| Updates online node map. |
| `aggregation_completed`| FL Server | Dashboard | `AggregationStatus`| Signals end of FedAvg. |

## Example Payload
```json
{
  "event": "metrics_updated",
  "data": {
    "experiment_id": "exp-883",
    "round_number": 5,
    "training_loss": 0.24,
    "dice_score": 0.82
  }
}
```
