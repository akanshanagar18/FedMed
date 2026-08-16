"""
CKKS encryption context management.

The "context" is TenSEAL's bundle of scheme parameters and keys
(public key, secret key, Galois keys, relinearization keys). Every
node that encrypts data needs a context; only the node(s) allowed to
decrypt should ever hold the version containing the secret key.
"""

import tenseal as ts

class ContextError(Exception):
    pass

DEFAULT_POLY_MODULUS_DEGREE = 8192
DEFAULT_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
DEFAULT_GLOBAL_SCALE = 2 ** 40


def create_ckks_context(
    poly_modulus_degree: int = DEFAULT_POLY_MODULUS_DEGREE,
    coeff_mod_bit_sizes=None,
    global_scale: float = DEFAULT_GLOBAL_SCALE,
    generate_galois_keys: bool = True,
) -> ts.Context:
    """Create a new CKKS context, including a freshly generated key pair."""
    coeff_mod_bit_sizes = coeff_mod_bit_sizes or DEFAULT_COEFF_MOD_BIT_SIZES

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = global_scale

    if generate_galois_keys:
        context.generate_galois_keys()

    return context


def validate_context(context: ts.Context) -> bool:
    """Confirm a context is usable for decryption. Raises ContextError otherwise."""
    if context is None:
        raise ContextError("Encryption context is None")
    if not context.is_private():
        raise ContextError(
            "Context does not contain a secret key; it cannot be used "
            "to decrypt. Did you receive a public-only context?"
        )
    return True


def make_public_context(context: ts.Context) -> ts.Context:
    """Strip the secret key from a copy of `context`."""
    validate_context(context)
    public_context = context.copy()
    public_context.make_context_public()
    return public_context


def serialize_context(context: ts.Context, save_secret_key: bool = False) -> bytes:
    try:
        return context.serialize(save_secret_key=save_secret_key)
    except Exception as exc:
        raise ContextError(f"Failed to serialize context: {exc}") from exc


def deserialize_context(data: bytes) -> ts.Context:
    try:
        return ts.context_from(data)
    except Exception as exc:
        raise ContextError(f"Failed to deserialize context: {exc}") from exc

if __name__ == "__main__":
    print("Creating CKKS context...")
    ctx = create_ckks_context()
    print("Validating context...")
    validate_context(ctx)
    print("Context is valid and private:", ctx.is_private())
    print("Creating public context...")
    pub_ctx = make_public_context(ctx)
    print("Public context is private:", pub_ctx.is_private())
    print("Code ran successfully.")