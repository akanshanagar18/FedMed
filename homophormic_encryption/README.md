# `privacy` — TenSEAL Homomorphic Encryption Module

Part of **FedMed**, a cross-silo federated learning engine for training
medical ML models (e.g. brain tumor segmentation) across hospitals
without ever moving raw patient data or raw model weights off-site.

This module implements the **encryption layer**: CKKS homomorphic
encryption via [TenSEAL](https://github.com/OpenMined/TenSEAL), so the
central server can aggregate model updates from multiple hospitals
while those updates remain mathematically encrypted the entire time.

## Why CKKS

CKKS supports approximate arithmetic (addition and multiplication) directly
on ciphertexts of real numbers, which is exactly what's needed for
averaging floating-point model weights. It trades exact precision for
speed and practicality — acceptable here since FedAvg-style aggregation
already tolerates small numerical noise.

## Module layout

```
privacy/
├── __init__.py          # public API surface
├── context.py            # CKKS context creation, key handling, validation
├── tensor_utils.py        # PyTorch tensor <-> flat list conversion
├── encryptor.py            # tensor / state_dict encryption
├── decryptor.py             # tensor / state_dict decryption
├── serialization.py          # encrypted data <-> bytes, for network transfer
├── aggregation.py              # encrypted addition, sum & weighted FedAvg
├── exceptions.py                 # module-specific exception hierarchy
└── tests/
    ├── test_encryption.py          # context + encrypt/decrypt round-trips
    └── test_aggregation.py           # encrypted sum & weighted aggregation
```

## Typical flow in FedMed

1. **Key holder** (e.g. a trusted coordinator) calls `create_ckks_context()`
   once and distributes a *public* context (`make_public_context`) to the
   central server and all hospital nodes; only it keeps the private context.
2. **Each hospital** trains locally, then calls `encrypt_state_dict(context,
   model.state_dict())` and `serialize_encrypted_state_dict(...)` before
   sending the result to the central server.
3. **Central server** calls `weighted_aggregate_encrypted_state_dicts(...)`
   (weighting by e.g. each hospital's local sample count) to homomorphically
   combine updates — it never decrypts anything.
4. **Key holder** (or a secure multi-party protocol) decrypts the aggregated
   result with `decrypt_state_dict(...)` and the new global model is
   distributed for the next training round.

## Quick example

```python
import torch
from privacy import (
    create_ckks_context,
    encrypt_state_dict,
    decrypt_state_dict,
    weighted_aggregate_encrypted_state_dicts,
)

context = create_ckks_context()  # holds the secret key

# --- at each hospital ---
hospital_state_dicts = [
    {"fc.weight": torch.randn(4, 4), "fc.bias": torch.randn(4)}
    for _ in range(3)
]
encrypted_updates = [encrypt_state_dict(context, sd) for sd in hospital_state_dicts]
sample_counts = [120, 340, 75]  # patients seen locally

# --- at the central server (never sees plaintext weights) ---
aggregated = weighted_aggregate_encrypted_state_dicts(encrypted_updates, sample_counts)

# --- back at the key holder ---
global_state_dict = decrypt_state_dict(aggregated)
```

## Error handling

All failures raise a subclass of `PrivacyModuleError`:

| Exception             | Raised when                                             |
|------------------------|----------------------------------------------------------|
| `ContextError`          | Context is missing, invalid, or lacks the secret key     |
| `EncryptionError`        | A tensor/state_dict fails to encrypt or serialize        |
| `DecryptionError`         | A ciphertext fails to decrypt                             |
| `AggregationError`         | Aggregation inputs are empty, mismatched, or invalid       |
| `ShapeMismatchError`         | Two encrypted parameters being combined differ in shape     |

## Running tests

```bash
pip install tenseal torch pytest
pytest privacy/tests/ -v
```

## Security notes

- The CKKS scheme is **approximate**: decrypted values will differ from
  the originals by a small epsilon (tests use `atol=1e-2`).
- Only distribute contexts containing the secret key to parties that are
  supposed to decrypt. Use `make_public_context()` for everyone else.
- Parameter *shapes* travel in the clear alongside ciphertexts — this
  module treats shape as non-sensitive metadata, not patient data.
