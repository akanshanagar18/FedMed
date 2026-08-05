import tenseal as ts

def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )

    context.global_scale = 2**40
    context.generate_galois_keys()

    return context