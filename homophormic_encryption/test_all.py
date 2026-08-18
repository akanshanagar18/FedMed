import sys
import os
import torch
sys.path.insert(0, os.path.abspath('..'))
from homophormic_encryption import create_ckks_context, encrypt_state_dict, decrypt_state_dict, weighted_aggregate_encrypted_state_dicts

def main():
    print("Creating context...")
    context = create_ckks_context()

    print("Setting up local hospital updates...")
    hospital_state_dicts = [
        {"fc.weight": torch.randn(4, 4), "fc.bias": torch.randn(4)}
        for _ in range(3)
    ]
    sample_counts = [120, 340, 75]

    from homophormic_encryption.serialization import serialize_encrypted_state_dict, deserialize_encrypted_state_dict

    print("Encrypting and serializing updates at hospitals...")
    encrypted_updates = [encrypt_state_dict(context, sd) for sd in hospital_state_dicts]
    serialized_updates = [serialize_encrypted_state_dict(eu) for eu in encrypted_updates]

    print(f"Network Transfer... (Bytes sent: {[len(su) for su in serialized_updates]})")

    print("Deserializing and aggregating updates at server...")
    deserialized_updates = [deserialize_encrypted_state_dict(context, su) for su in serialized_updates]
    aggregated = weighted_aggregate_encrypted_state_dicts(deserialized_updates, sample_counts)

    print("Decrypting aggregated result...")
    global_state_dict = decrypt_state_dict(aggregated)

    print("Shapes of decrypted tensors:")
    for k, v in global_state_dict.items():
        print(f"{k}: {v.shape}")

    print("All tests passed successfully.")

if __name__ == "__main__":
    main()
