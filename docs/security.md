# FedMed v2.0 Production TLS & Zero Trust Security Specification

## 1. TLS Architecture Overview

FedMed v2.0 enforces end-to-end transport layer security (TLS 1.3 / mTLS) for all gRPC communication between hospital client nodes (`hospital_alpha`, `hospital_beta`, `hospital_gamma`) and the central Flower orchestrator server.

```
+--------------------------+                         +--------------------------+
|  Hospital Client Node    |                         |  Central Flower Server   |
|                          |     mTLS (gRPC)         |                          |
|  - Client Cert (X.509)   | <=====================> |  - Server Cert (SAN)     |
|  - Client Key (RSA 2048) |   Port 8080 (Encrypted) |  - Root CA Verification  |
+--------------------------+                         +--------------------------+
```

---

## 2. Certificate Hierarchy & Automatic Local Generation

FedMed includes an automated X.509 certificate authority generator (`privacy/tls_cert_gen.py`) powered by Python's `cryptography` library. No manual OpenSSL invocations are required.

```
                     +---------------------------+
                     |    FedMed Root CA         |
                     |  (certs/ca.crt, ca.key)   |
                     +---------------------------+
                                   |
                +------------------+------------------+
                |                                     |
                v                                     v
+-------------------------------+   +-------------------------------+
|  FedMed Server Certificate    |   |  FedMed Client Certificate    |
| (certs/server.crt, server.key)|   | (certs/client.crt, client.key)|
| SAN: localhost, 127.0.0.1     |   | ExtendedKeyUsage: ClientAuth  |
+-------------------------------+   +-------------------------------+
```

---

## 3. Mutual Authentication (mTLS)

1. **Server Verification**: Clients verify the server certificate against the shared Root CA (`ca.crt`).
2. **Client Verification**: The server verifies client identities during gRPC handshake against the Root CA.
3. **SAN Validation**: Subject Alternative Names (`localhost`, `127.0.0.1`, `::1`) protect against Man-in-the-Middle (MitM) domain spoofing.

---

## 4. Threat Model & Defense-in-Depth

| Threat Vector | Defense Mechanism | Mitigation Layer |
| :--- | :--- | :--- |
| **Network Eavesdropping** | TLS 1.3 Encryption | gRPC Transport Layer |
| **Server Impersonation** | X.509 SAN Certificate Validation | TLS Handshake |
| **Model Inversion Attacks** | Sample-level Differential Privacy (Opacus) | Local Training Engine |
| **Parameter Reconstruction** | TenSEAL CKKS Homomorphic Encryption | Server Aggregation Engine |
