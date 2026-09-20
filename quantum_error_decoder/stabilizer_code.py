# stabilizer_code.py

import numpy as np

from qec.symplectic import (
    gf2_row_basis,
    is_in_span,
    pauli_to_symplectic,
    stabilizers_commute,
    syndrome,
)


class StabilizerCode:
    
    # Constructor
    def __init__(self, Mx, Mz, logical_x=None, logical_z=None):
        self.Mx = np.array(Mx, dtype=np.uint8)
        self.Mz = np.array(Mz, dtype=np.uint8)

        if self.Mx.shape != self.Mz.shape:
            raise ValueError("Mx and Mz must have the same shape")
        if self.Mx.ndim != 2:
            raise ValueError("Mx and Mz must be matrices")

        self.logical_x = logical_x
        self.logical_z = logical_z

        self.n = self.Mx.shape[1]

        # Check whether it's a valid stabilizer code
        if not stabilizers_commute(self.Mx, self.Mz):
            raise ValueError("Stabilizer generators do not commute")

        # Put each generator in (x | z) form once.
        self.M = np.concatenate([self.Mx, self.Mz], axis=1)

        # Save a basis now, rather than row-reducing for every error we test.
        self.stabilizer_basis, self.stabilizer_pivots = gf2_row_basis(self.M)

        # r is the number of independent stabilizer generators.
        self.r = len(self.stabilizer_basis)
        self.k = self.n - self.r

    # Alternative constructor - use pauli strings
    @classmethod
    def from_pauli_strings(cls, stabilizers, logical_x=None, logical_z=None):
        rows = [pauli_to_symplectic(stabilizer) for stabilizer in stabilizers]

        Mx = np.array([x for x, z in rows], dtype=np.uint8)
        Mz = np.array([z for x, z in rows], dtype=np.uint8)

        return cls(Mx, Mz, logical_x=logical_x, logical_z=logical_z)

    # ---- Mehods ----
    def syndrome(self, x, z):
        return syndrome(self.Mx, self.Mz, x, z)
    
    def is_stabilizer(self, x, z):
        pauli = np.concatenate([x, z])

        # A stabilizer is an XOR combination of generator rows.
        return is_in_span(self.stabilizer_basis, self.stabilizer_pivots, pauli)
    
    # # ---- Pauli string versions---- 
    def syndrome_for_pauli(self, pauli):
        x, z = pauli_to_symplectic(pauli)
        return self.syndrome(x, z)
    
    def is_stabilizer_for_pauli(self, pauli):
        x, z = pauli_to_symplectic(pauli)
        return self.is_stabilizer(x, z)
