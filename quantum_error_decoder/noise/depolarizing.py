# noise/depolarizingnoise.py

import numpy as np

from quantum_error_decoder.noise.base import NoiseModel

# Noise model: Independent single-qubit depolarizing noise
# (1) Each qubit (independently) has error with probability p --- no error with probability 1 - p
# (2) If qubit has error ---> error is chosen uniformly to be X, Y or Z

# Each qubit independently experiences:
# I     1-p
# X     p/3
# Y     p/3
# Z     p/3

class DepolarizingNoise(NoiseModel):
    def __init__(self, p):
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")

        self.p = p

    # Sample an n-qubit Pauli error in symplectic form.
    def sample(self, n, rng):
        x = np.zeros(n, dtype=np.uint8)
        z = np.zeros(n, dtype=np.uint8)

        # Independently mark each qubit as erroneous with probability p
        error_mask = rng.random(n) < self.p     # [ True, False, False, True, ...]

        # Uniformly choose X, Y, or Z for each qubit
        paulis = rng.integers(0, 3, size=n)
        
        # X-error: For qubit that have (error AND X or Y)
        # Z-error: For qubit that have (error AND Z or Y)
        mask_x = error_mask & ((paulis == 0) | (paulis == 1))
        mask_z = error_mask & ((paulis == 1) | (paulis == 2))
        x[mask_x] = 1
        z[mask_z] = 1
        return x, z
