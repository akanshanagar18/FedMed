# Privacy Module Guide

**Owners:** Akansha & Rakshit
**Folder:** `privacy/`

## Objective
Provide cryptographic utilities to ensure patient data remains perfectly secure. This module wraps TenSEAL to provide Homomorphic Encryption (CKKS) capabilities for the model weights.

## Dependencies
- **None.** This module is a pure cryptographic utility layer.

## Integration Points
- **Expected Inputs:** 1D or flattened float arrays/tensors (Model Weights).
- **Expected Outputs:** TenSEAL CKKS vectors (Ciphertexts).

## Development Checklist
- [ ] Implement Context generation (Keys).
- [ ] Implement `encrypt_weights()` and `decrypt_weights()` wrappers.
- [ ] (Future) Add differential privacy noise generators.

## Common Mistakes to Avoid
- **Do not** assume specific tensor shapes. Ensure your wrappers can flatten and reshape arrays seamlessly.
- **Beware** of polynomial degree parameters. High degrees cause massive memory spikes. Expose these as configurable settings.
