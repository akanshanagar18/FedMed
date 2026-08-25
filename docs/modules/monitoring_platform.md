# Monitoring Platform Module Guide

**Owner:** Siddhant
**Folder:** `dashboard/`

## Objective
Provide a unified FastAPI backend and React frontend to visualize federated training progress in real-time. This module provides the central repository contracts.

## Dependencies
- **Shared Contracts (`common/contracts/`)**: For schema validation.

## Integration Points
- **Expected Inputs:** REST `POST` requests from the FL Server containing metrics and status.
- **Expected Outputs:** Live UI charts, WebSocket broadcasts to frontend clients.

## Development Checklist
- [ ] Finalize SQLAlchemy models for metric persistence.
- [ ] Build the WebSocket manager for multi-client broadcasting.
- [ ] Build React frontend using Tailwind and Recharts.

## Common Mistakes to Avoid
- **Do not** tightly couple API routes to business logic. Use the Service layers (`app/services/`).
- **Do not** change schemas unilaterally. Remember that the FL Server team (Vishnu) relies on `common/contracts/` to send you data.
