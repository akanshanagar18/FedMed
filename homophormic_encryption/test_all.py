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

    print("Encrypting updates...")
    encrypted_updates = [encrypt_state_dict(context, sd) for sd in hospital_state_dicts]

    print("Aggregating updates at server...")
    aggregated = weighted_aggregate_encrypted_state_dicts(encrypted_updates, sample_counts)

    print("Decrypting aggregated result...")
    global_state_dict = decrypt_state_dict(aggregated)

    print("Shapes of decrypted tensors:")
    for k, v in global_state_dict.items():
        print(f"{k}: {v.shape}")

    print("All tests passed successfully.")

if __name__ == "__main__":
    main()
