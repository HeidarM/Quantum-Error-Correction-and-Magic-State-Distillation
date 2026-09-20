# quantum_error_decoder/experiments/fixed_weight_errors.py
# Run from the repository root as: python -m quantum_error_decoder.experiments.fixed_weight_errors

from itertools import combinations, product

from quantum_error_decoder.codes.planar_surface import planar_surface_code
from quantum_error_decoder.codes.rotated_surface_d3 import rotated_surface_d3
from quantum_error_decoder.codes.shor9 import shor9
from quantum_error_decoder.decoders.lookup import LookupDecoder
from qec.symplectic import pauli_to_symplectic, symplectic_to_pauli


def all_k_qubit_errors(n, k):
    for qubits in combinations(range(n), k):
        for paulis in product("XYZ", repeat=k):
            error = ["I"] * n

            for q, p in zip(qubits, paulis):
                error[q] = p
                
            # For iterator: return the next error one at a time (for memory efficiency vs full list at once)
            yield "".join(error)


def fixed_weight_results(code, decoder, max_weight=3, verbose=False):
    results = {}

    # Test every Pauli error with exactly k non-identity factors.
    for k in range(1, min(max_weight, code.n) + 1):
        successes = 0
        num_k_errors = 0

        for error in all_k_qubit_errors(code.n, k):
            error_x, error_z = pauli_to_symplectic(error)
            syndrome = code.syndrome(error_x, error_z)
            correction_x, correction_z = decoder.decode(syndrome)
            residual_x = error_x ^ correction_x
            residual_z = error_z ^ correction_z
            success = code.is_stabilizer(residual_x, residual_z)

            successes += success
            num_k_errors += 1

            if verbose:
                correction = symplectic_to_pauli(correction_x, correction_z)
                residual = symplectic_to_pauli(residual_x, residual_z)
                print(
                    f"k={k}: {error} | syndrome {syndrome} -> "
                    f"correction {correction} | residual {residual} | success {success}"
                )

        failures = num_k_errors - successes

        results[k] = {
            "successes": successes,
            "failures": failures,
            "num_k_errors": num_k_errors,
            "success_rate": successes / num_k_errors,
        }

    return results


def print_fixed_weight_results(label, results):
    print(f"\n{label}")
    print("-" * len(label))

    for k, result in results.items():
        count = f"{result['successes']}/{result['num_k_errors']}"
        success_rate = 100 * result["success_rate"]

        print(
            f"k={k}: {count:^14}"
            f"\tP(success | wt(E) = {k})   =  {success_rate:5.1f}%"
        )


if __name__ == "__main__":
    planar_surface_d3 = planar_surface_code(3)
    codes_to_test = [
        (shor9, "Shor-9"),
        (rotated_surface_d3, "Rotated surface-d3"),
        (planar_surface_d3, "Planar surface-d3"),
    ]

    for code, label in codes_to_test:
        decoder = LookupDecoder(code)
        results = fixed_weight_results(code, decoder, max_weight=3)
        print_fixed_weight_results(label, results)
