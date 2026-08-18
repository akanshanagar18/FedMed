Privacy Layer (FedMed)

Overview
The privacy package orchestrates encrypted model-update transmission and secure aggregation.
It delegates all low-level homomorphic encryption operations to the partner homomorphic_encryption package.

Flow
Client:
  - Encrypt model updates using homomorphic_encryption.encryptor
  - Serialize encrypted updates via privacy.communication.serialize_update
  - Send serialized payload to server

Server:
  - Receive serialized payloads
  - Deserialize via privacy.communication.deserialize_update (server supplies public TenSEAL context when required)
  - Call SecureAggregator.aggregate(serialized_updates, weights)
  - Send serialized aggregated encrypted update back to clients

Client (receive):
  - Deserialize and decrypt using the secret TenSEAL context via partner decryptor
  - Apply global update to local model

Public API (privacy package)
  - serialize_update(update: Any) -> bytes
  - deserialize_update(data: bytes, context: Optional[ts.Context] = None) -> Any
  - aggregate_encrypted_updates(encrypted_updates: Iterable[Dict[str, Any]], weights: Optional[List[float]] = None) -> Dict[str, Any]
  - SecureAggregator(context: Optional[Any] = None).aggregate(serialized_updates: List[bytes], weights: Optional[List[float]] = None) -> bytes
  - Benchmark helpers in privacy.benchmark

Expected partner homomorphic_encryption API
The privacy layer expects the partner package to provide the following functions/modules (names and signatures used by privacy code):

  homomorphic_encryption.context
    - create_ckks_context(...) -> ts.Context
    - make_public_context(context: ts.Context) -> ts.Context
    - serialize_context(context: ts.Context, save_secret_key: bool = False) -> bytes
    - deserialize_context(data: bytes) -> ts.Context

  homomorphic_encryption.encryptor
    - encrypt_state_dict(context: ts.Context, state_dict: Dict[str, torch.Tensor]) -> Dict[str, EncryptedParam]

  homomorphic_encryption.decryptor
    - decrypt_state_dict(encrypted_state_dict: Dict[str, EncryptedParam]) -> Dict[str, torch.Tensor]

  homomorphic_encryption.serialization
    - serialize_encrypted_state_dict(encrypted_state_dict: Dict[str, EncryptedParam]) -> bytes
    - deserialize_encrypted_state_dict(context: ts.Context, data: bytes) -> Dict[str, EncryptedParam]
    or generic alternatives:
    - serialize(obj) -> bytes
    - deserialize(bytes) -> obj

  homomorphic_encryption.aggregation
    - aggregate_encrypted_state_dicts(list_of_state_dicts: List[Dict[str, EncryptedParam]]) -> Dict[str, EncryptedParam]
    - weighted_aggregate_encrypted_state_dicts(list_of_state_dicts: List[Dict[str, EncryptedParam]], weights: List[float]) -> Dict[str, EncryptedParam]
    or lower-level alternative:
    - aggregate_ciphertexts(list_of_ciphertexts, weights: Optional[List[float]] = None) -> ciphertext

Notes on weighted aggregation
If homomorphic_encryption provides weighted_aggregate_encrypted_state_dicts, the privacy layer will use it.
If server-side weighted aggregation is not supported by the partner API, apply weighting client-side (scale plaintexts before encryption) to avoid performing ciphertext*scalar operations on the server.

Testing
Integration tests are provided under privacy/tests and are skipped when homomorphic_encryption is not importable.
Important tests:
  - Encrypt/decrypt correctness (roundtrip)
  - Encrypted addition and aggregation
  - Weighted aggregation for multiple clients
  - Shape mismatch detection
  - Empty aggregation error handling
  - Serialization roundtrip
  - End-to-end pipeline using SecureAggregator and public context

Benchmarking
privacy/benchmark.py contains lightweight helpers:
  - time_serialization_roundtrip(update, rounds)
  - time_aggregation(serialized_updates, weights)
Use these to compare plain vs encrypted flows and to measure serialization and orchestration overhead. Do not include benchmark helpers in production control paths.

Security and limitations
  - The server must not hold secret keys. The server should be given a public TenSEAL context (make_context_public) that allows ciphertext reconstruction but not decryption.
  - The privacy layer never decrypts on the server side.
  - The fallback pickle-based serialization in privacy.communication is for local development only. Do not use pickle over untrusted networks. Replace with partner serialization for production.
  - CKKS is approximate. Tests use tolerances rather than exact equality.
  - The privacy layer does not attempt to validate internal TenSEAL context compatibility beyond what the partner package exposes. If required, add partner-provided validation helpers.

Integration checklist before production
  - Ensure homomorphic_encryption package is installed and importable.
  - Verify the partner exposes the expected function names and signatures.
  - Run privacy integration tests under privacy/tests with the real TenSEAL-backed partner implementation.
  - Validate weighted aggregation semantics match FedMed expectations (average vs sum).
  - Confirm network transports treat serialized blobs as opaque and do not attempt to inspect ciphertext internals.

Contact
If the partner API uses different names, update privacy.communication and privacy.aggregation to match the actual exported functions.