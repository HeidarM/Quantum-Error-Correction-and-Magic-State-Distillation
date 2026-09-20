# magic_state_distillation/distillation.py

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from magic_state_distillation.quantum_reed_muller_code import H_X, H_Z, reed_muller_encoder
from magic_state_distillation.state_preparation import prepare_encoded_plus_block

# Using the 15-qubit Quantum Reed-Muller code [[15, 1, 3]]
# 15-to-1 Bravyi-Kitaev magic state distillation protocol


# Create a magic state on qubit q. The noise model makes it imperfect in simulation.
def magic_state(qc, q):
    qc.h(q)
    qc.t(q)
    return qc


# Consume one noisy |T> resource state to apply a noisy T gate to a data qubit.
def inject_t_gate(qc, data_qubit, resource_qubit, resource_measurement):
    # (1) Prepare the imperfect |T> = T|+> resource state.
    magic_state(qc, resource_qubit)

    # (2) Teleport T onto the data qubit.
    # Measurement 0 gives T
    # Measurement 1 gives T-dagger ---> corrected by S.
    qc.cx(data_qubit, resource_qubit)
    qc.measure(resource_qubit, resource_measurement)

    with qc.if_test((resource_measurement, 1)):
        qc.s(data_qubit)

    # (3) The resource has been consumed; reset it before its next use.
    qc.reset(resource_qubit)
    return qc


# Measure all Reed-Muller stabilizers serially.
def measure_code_syndrome(qc, data_register, resource_qubit):
    syndrome = ClassicalRegister(H_X.shape[0] + H_Z.shape[0], "syndrome")
    qc.add_register(syndrome)

    for i in range(H_X.shape[0]):
        # Prepare the ancilla in |+> to measure an X-type Pauli product.
        qc.h(resource_qubit)

        for q in range(H_X.shape[1]):
            if H_X[i, q]:
                qc.cx(resource_qubit, data_register[q])

        qc.h(resource_qubit)
        qc.measure(resource_qubit, syndrome[i])
        qc.reset(resource_qubit)

    for i in range(H_Z.shape[0]):
        # The ancilla starts in |0> to measure a Z-type Pauli product.
        for q in range(H_Z.shape[1]):
            if H_Z[i, q]:
                qc.cx(data_register[q], resource_qubit)

        qc.measure(resource_qubit, syndrome[H_X.shape[0] + i])
        qc.reset(resource_qubit)

    return syndrome


# Magic-state distillation circuit: encode |+>_L, then apply 15 noisy T injections.
def magic_state_distillation_circuit():
    
    # Encode |+>_L in the [[15, 1, 3]] Quantum Reed-Muller code
    # qc = prepare_encoded_plus_block(H_Z)
    data_register = QuantumRegister(15, "data")
    resource_register = QuantumRegister(1, "anc")
    qc = QuantumCircuit(data_register, resource_register)

    for q in range(5):
        qc.h(data_register[q])
    qc.compose(reed_muller_encoder(), qubits=data_register, inplace=True)

    # Registers
    injection = ClassicalRegister(1, "injection")
    qc.add_register(injection)

    # Using resource magic states to act with T^15
    # The [[15,1,3]] quantum Reed–Muller code has special structure: triorthogonality.
    # Due to this we have Transversal non-Clifford action: T_L = T^15.
    for q in range(15):
        inject_t_gate(qc, data_register[q], resource_register[0], injection[0])

    # The encoded |T>_L = T_L |+>_L is noisy, so measure the code syndrome.
    syndrome = measure_code_syndrome(qc, data_register, resource_register[0])

    # Decode only accepted blocks. The output |T> is on data_register[0].
    with qc.if_test((syndrome, 0)):
        qc.compose(reed_muller_encoder().inverse(), qubits=data_register, inplace=True)

    return qc
