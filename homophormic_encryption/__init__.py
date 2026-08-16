if __name__ == "__main__":
    # If you run this file directly from your IDE, it will run the test script
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from homophormic_encryption import test_all
    print("Running test_all.py since __init__.py was executed directly...\n")
    test_all.main()
else:
    from .context import create_ckks_context, make_public_context, validate_context, serialize_context, deserialize_context
    from .encryptor import encrypt_parameter, encrypt_state_dict
    from .decryptor import decrypt_parameter, decrypt_state_dict
    from .serialization import serialize_encrypted_state_dict, deserialize_encrypted_state_dict
    from .aggregation import add_encrypted, aggregate_encrypted_state_dicts, weighted_aggregate_encrypted_state_dicts
    from .exceptions import PrivacyModuleError, ContextError, EncryptionError, DecryptionError, AggregationError, ShapeMismatchError
