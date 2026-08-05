import tenseal as ts

def encrypt_weights(context, weights):

    encrypted = ts.ckks_vector(context, weights)

    return encrypted