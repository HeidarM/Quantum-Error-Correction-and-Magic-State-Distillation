# tests/test_shor9.py
# Run from the repository root as: python -m quantum_error_decoder.tests.test_shor9

from quantum_error_decoder.codes.shor9 import shor9
from quantum_error_decoder.decoders.lookup import LookupDecoder
from qec.symplectic import multiply_paulis, pauli_to_symplectic


def one_qubit_paulis(n):
    for qubit in range(n):
        for pauli in "XYZ":
            error = ["I"] * n
            error[qubit] = pauli
            yield "".join(error)


if __name__ == "__main__":
        
    decoder = LookupDecoder(shor9)
    errors = one_qubit_paulis(shor9.n)
    
    decoder.print_table()
    print()

    for error in errors:
        syndrome = shor9.syndrome_for_pauli(error)
        correction = decoder.decode_to_pauli(syndrome)
        residual = multiply_paulis(error, correction)
        residual_syndrome = shor9.syndrome_for_pauli(residual)

        x, z = pauli_to_symplectic(error)
        print("error:             ", error)
        print("x, z:              ", x, z)
        print("syndrome:          ", syndrome)
        print("correction:        ", correction)
        print("residual:          ", residual)
        print("residual syndrome: ", residual_syndrome)
        print()
        
