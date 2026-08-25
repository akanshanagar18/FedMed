"""
Module: privacy.context

Purpose:
Production TenSEAL CKKS Homomorphic Encryption Context Management.
Creates, configures, and serializes CKKS encryption contexts with public/private key separation.
"""

import logging
from typing import Optional
import tenseal as ts

logger = logging.getLogger(__name__)


def create_ckks_context(
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: Optional[list] = None,
    global_scale: Optional[float] = None,
) -> ts.Context:
    """
    Creates and configures a production TenSEAL CKKS homomorphic encryption context.
    """
    if coeff_mod_bit_sizes is None:
        if poly_modulus_degree == 4096:
            coeff_mod_bit_sizes = [40, 20, 40]
        elif poly_modulus_degree == 2048:
            coeff_mod_bit_sizes = [30, 20, 30]
        else:
            coeff_mod_bit_sizes = [60, 40, 40, 60]

    if global_scale is None:
        if poly_modulus_degree == 4096:
            global_scale = 2**20
        elif poly_modulus_degree == 2048:
            global_scale = 2**16
        else:
            global_scale = 2**40

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = global_scale
    context.generate_galois_keys()
    context.generate_relin_keys()

    return context


create_context = create_ckks_context



def get_public_context(context: ts.Context) -> ts.Context:
    """
    Returns a copy of the context with secret key dropped for public evaluation/aggregation.
    """
    public_context = context.copy()
    public_context.make_context_public()
    return public_context



def serialize_context(context: ts.Context, save_secret_key: bool = True) -> bytes:
    """
    Serializes TenSEAL context to bytes string.
    """
    return context.serialize(save_secret_key=save_secret_key)


def deserialize_context(context_bytes: bytes) -> ts.Context:
    """
    Deserializes bytes string back into a TenSEAL Context.
    """
    return ts.context_from(context_bytes)