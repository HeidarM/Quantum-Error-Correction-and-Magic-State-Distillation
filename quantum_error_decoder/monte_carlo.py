# quantum_error_decoder/monte_carlo.py

import numpy as np


# Code-capacity Monte Carlo estimate of the logical error rate:
# one random data-error pattern per shot, with perfect syndrome extraction.
#
# Each shot:
# 1. Sample physical Pauli error E.
# 2. Compute syndrome s(E) and decoder correction R(s).
# 3. Form residual E + R over F_2.
# 4. Count failure when the residual is not a stabilizer (a logical Pauli).
def estimate_logical_failure_rate(
    code,
    decoder,
    noise,
    shots=10000,
    seed=7,
):
    rng = np.random.default_rng(seed)
    failures = 0

    for _ in range(shots):
        # Sample physical error from the noise model.
        error_x, error_z = noise.sample(code.n, rng)

        # Decode the syndrome.
        syndrome = code.syndrome(error_x, error_z)
        correction_x, correction_z = decoder.decode(syndrome)

        # Residual = physical error + correction over F_2.
        residual_x = error_x ^ correction_x
        residual_z = error_z ^ correction_z

        # A non-stabilizer residual is a logical failure.
        if not code.is_stabilizer(residual_x, residual_z):
            failures += 1

    return failures / shots
