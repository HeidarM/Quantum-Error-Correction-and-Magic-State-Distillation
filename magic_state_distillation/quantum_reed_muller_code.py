# magic_state_distillation/quantum_reed_muller_code.py

import numpy as np
from itertools import combinations
from qiskit import QuantumCircuit
from qec.symplectic import gf2_row_basis, is_in_span

# Using the 15-qubit Quantum Reed-Muller code [[15, 1, 3]]

# X-stabilizers
H_X  = np.array([
    [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
    [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
    [0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
    [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
], dtype=np.uint8)

# Z-stabilizers
H_Z = np.array([
    [0,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,1,1,0,1,0,0,0,0,0,0,0,0,0],
    [1,1,0,1,0,0,1,0,0,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,1,1,0,0,0,0,0,0],
    [1,0,1,0,0,0,0,1,0,1,0,0,0,0,0],
    [1,1,0,0,0,0,0,1,0,0,1,0,0,0,0],
    [1,1,1,1,0,0,0,1,0,0,0,1,0,0,0],
    [1,0,0,1,0,0,0,1,0,0,0,0,1,0,0],
    [0,1,0,1,0,0,0,1,0,0,0,0,0,1,0],
    [0,0,1,1,0,0,0,1,0,0,0,0,0,0,1],
], dtype=np.uint8)


# Logical operators
LOGICAL_X = np.ones(15, dtype=np.uint8)
LOGICAL_Z = np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.uint8)

# General unitary encoder for the [[15, 1, 3]] Reed-Muller code.
def reed_muller_encoder():
    # Input convention:
    # q0      = arbitrary logical input
    # q1..q4  = |+> ancillas for the four X stabilizers
    # q5..q14 = |0> ancillas
    #
    # The encoder maps X(q0) to logical X and X(q1..q4) to H_X.

    # The first five columns specify logical X and the four X stabilizers.
    columns = [LOGICAL_X, *H_X]
    basis, pivots = gf2_row_basis(columns)

    # Complete them to an invertible binary 15 by 15 matrix.
    for qubit in range(15):
        if len(columns) == 15:
            break

        standard_basis_vector = np.zeros(15, dtype=np.uint8)
        standard_basis_vector[qubit] = 1

        if not is_in_span(basis, pivots, standard_basis_vector):
            columns.append(standard_basis_vector)
            basis, pivots = gf2_row_basis(columns)

    matrix = np.column_stack(columns)
    working_matrix = matrix.copy()
    elimination_operations = []

    # Reduce matrix to the identity with row operations over GF(2).
    for column in range(15):
        pivot = next(
            (row for row in range(column, 15) if working_matrix[row, column]),
            None,
        )
        if pivot is None:
            raise ValueError("Encoder matrix is not invertible")

        if pivot != column:
            working_matrix[[column, pivot]] = working_matrix[[pivot, column]]
            elimination_operations.append(("swap", column, pivot))

        for row in range(15):
            if row != column and working_matrix[row, column]:
                working_matrix[row] ^= working_matrix[column]
                elimination_operations.append(("cx", column, row))

    # Reverse the elimination operations to implement |x> -> |matrix x>.
    encoder = QuantumCircuit(15, name="RM_encoder")

    # Swap the logical |0> and |1> labels so transversal T^15 is logical T.
    encoder.x(0)

    for operation, control, target in reversed(elimination_operations):
        if operation == "swap":
            encoder.swap(control, target)
        else:
            encoder.cx(control, target)

    return encoder



# Lookup table for decoding syndrome s ---> error e
def make_syndrome_lookup(H_Z):
    r, n = H_Z.shape
    num_syndromes = 2**r

    lookup = {}

    # Search corrections in increasing Hamming weight
    for weight in range(n + 1):
        for qubits in combinations(range(n), weight):

            e = np.zeros(n, dtype=np.uint8)
            e[list(qubits)] = 1

            # syndrome = H_Z @ e  mod 2
            s = (H_Z @ e) % 2

            # Convert binary array to integer
            s = sum(
                int(bit) << i
                for i, bit in enumerate(s)
            )

            # First solution found is minimum weight
            if s not in lookup:
                lookup[s] = list(qubits)

            if len(lookup) == num_syndromes:
                return lookup

    return lookup



def print_code_checks():
    print("H_Z H_X^T = 0  (mod 2)    ---->    Every Z stabilizer commutes with every X stabilizer")
    print((H_Z @ H_X.T) % 2)

    print()
    print("H_X z_L = 0  (mod 2)")
    print((H_X @ LOGICAL_Z) % 2)

    print("H_Z x_L = 0  (mod 2)")
    print((H_Z @ LOGICAL_X) % 2)
    print("Logical X and logical Z commute with all stabilizers.\n")

    print("x_L . z_L = 1  (mod 2)    ---->    Logical X and logical Z anticommute")
    print((LOGICAL_X @ LOGICAL_Z) % 2)


def apply_logical_x(qc):
    for q in np.flatnonzero(LOGICAL_X):
        qc.x(int(q))
    return qc


def apply_logical_z(qc):
    for q in np.flatnonzero(LOGICAL_Z):
        qc.z(int(q))
    return qc
