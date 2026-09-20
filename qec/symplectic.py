# symplectic.py

import numpy as np


def pauli_to_symplectic(pauli):
    # pauli = "IZZIZXXY"
    n = len(pauli)

    x = np.zeros(n, dtype=np.uint8)
    z = np.zeros(n, dtype=np.uint8)

    for i, p in enumerate(pauli):
        if p == "I":
            continue
        elif p == "X":
            x[i] = 1
        elif p == "Z":
            z[i] = 1
        elif p == "Y":
            x[i] = 1
            z[i] = 1
        else:
            raise ValueError(f"Invalid Pauli: {p}")

    return x, z


def symplectic_to_pauli(x, z):
    pauli = []

    for xi, zi in zip(x, z):
        if xi == 0 and zi == 0:
            pauli.append("I")
        elif xi == 1 and zi == 0:
            pauli.append("X")
        elif xi == 0 and zi == 1:
            pauli.append("Z")
        else:
            pauli.append("Y")

    return "".join(pauli)


def multiply_paulis(first, second):
    # Multiplying while ignoring overall phase with xor
    # "XII" x "XII" --> "III"
    # "XII" x "ZII" --> "YII"
    x1, z1 = pauli_to_symplectic(first)
    x2, z2 = pauli_to_symplectic(second)
    return symplectic_to_pauli(x1 ^ x2, z1 ^ z2) # a + b is a^b is


def symplectic_product(x1, z1, x2, z2):
    return (np.dot(x1, z2) + np.dot(z1, x2)) % 2


def syndrome(Mx, Mz, x, z):
    # Mx.z + Mz.x
    return (Mx @ z + Mz @ x) % 2


# Check whether generators commute
def stabilizers_commute(Mx, Mz):
    n_stabilizers = len(Mx)

    for i in range(n_stabilizers):
        for j in range(i + 1, n_stabilizers):
            if symplectic_product(
                Mx[i], Mz[i],
                Mx[j], Mz[j]
            ):
                return False

    # All generators commute
    return True


def gf2_rank(A):
    basis, pivots = gf2_row_basis(A)
    return len(basis)


# Input: A is a binary matrix over F_2.
# Return independent rows B with rowspan(B) = rowspan(A), and their pivots.
def gf2_row_basis(A):
    A = np.array(A, dtype=np.uint8, copy=True)

    n_rows, n_cols = A.shape
    rank = 0
    pivots = []

    for col in range(n_cols):
        pivot = None

        # Find a row with a 1 in this column to use as the pivot row.
        for row in range(rank, n_rows):
            if A[row, col]:
                pivot = row
                break

        if pivot is None:
            continue

        # Move this row into the next basis position.
        A[[rank, pivot]] = A[[pivot, rank]]

        # Over F_2, XOR means addition: row <- row + pivot_row.
        # This clears the pivot column in every other row.
        for row in range(n_rows):
            if row != rank and A[row, col]:
                A[row] ^= A[rank] # in F_2  (a+b is a^b)

        pivots.append(col)
        rank += 1

        if rank == n_rows:
            break

    return A[:rank], pivots


# Test whether vector v lies in rowspan(basis).
def is_in_span(basis, pivots, vector):
    vector = np.array(vector, dtype=np.uint8, copy=True)

    for row, pivot in zip(basis, pivots):
        if vector[pivot]:
            # v_pivot = 1, so cancel it: v <- v + row.
            vector ^= row

    # v = 0 means the original vector was a sum of basis rows.
    return not vector.any()
