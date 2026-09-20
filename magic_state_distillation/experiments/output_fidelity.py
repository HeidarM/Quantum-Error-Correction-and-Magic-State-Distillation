# magic_state_distillation/experiments/output_fidelity.py
# Run from the repository root as: python -m magic_state_distillation.experiments.output_fidelity

import numpy as np

from qiskit.quantum_info import Statevector, partial_trace, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

from magic_state_distillation.distillation import magic_state_distillation_circuit
from magic_state_distillation.quantum_reed_muller_code import H_X, H_Z, print_code_checks


# Estimate the output-state fidelity using accepted distillation attempts only.
def test_magic_state_distillation(p, accepted_target):
    accepted_attempts = 0
    total_attempts = 0
    total_fidelity = 0.0

    qc = magic_state_distillation_circuit()
    qc.save_statevector()
    exact_t = Statevector(np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2))

    magic_state_error = pauli_error([
        ("I", 1 - p),
        ("X", p / 3),
        ("Y", p / 3),
        ("Z", p / 3),
    ])
    noise_model = NoiseModel()
    noise_model.add_quantum_error(magic_state_error, "t", [15])
    simulator = AerSimulator(method="statevector", noise_model=noise_model)

    while accepted_attempts < accepted_target:
        result = simulator.run(qc, shots=1).result()
        outcome = next(iter(result.get_counts()))
        total_attempts += 1

        # syndrome is the left-most register in the Qiskit count key.
        if outcome.split()[0] != "0" * (H_X.shape[0] + H_Z.shape[0]):
            continue

        final_state = result.data(0)["statevector"]
        output_state = partial_trace(final_state, list(range(1, 16)))
        total_fidelity += state_fidelity(output_state, exact_t)
        accepted_attempts += 1

    return total_attempts, total_fidelity / accepted_attempts


if __name__ == "__main__":
    # print_code_checks()
    print()

    accepted_target = 100
    for p in [0.05, 0.08, 0.10, 0.15, 0.20]:

        print()
        print(f"Output-state fidelity test: p = {p}")
        total_attempts, average_fidelity = test_magic_state_distillation(p, accepted_target)
        print(f"accepted attempts: {accepted_target} out of {total_attempts}")
        print(f"average fidelity with exact |T>: {average_fidelity}")
