# magic_state_distillation/state_preparation.py

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from magic_state_distillation.quantum_reed_muller_code import make_syndrome_lookup


# Prepare/encode logical |+>_L by measuring and correcting Z stabilizers.
# Not unitary since it uses measurements
def prepare_encoded_plus_block(H_Z):
    # Logic: We want a state that is the unique state satisfying
    #   (A) in the stabilizer code: S_Z |+>_L = S_X |+>_L = |+>_L, and
    #   (B) Eigenstate: X_L |+>_L = |+>_L.
    # How to get it:
    #   (1) Prepare a state in |+>^n, then (B) is satisfied since X_L = products of X's in CSS code.
    #       Similarly S_X = 1 is satisfied, But it is not in the code state since S_Z != 1.
    #   (2) We measure Z-syndromes and project down to definite syndrome sectors.
    #   (3) We apply an X corrections, moving the state to the +1 sector of every S_Z stabilizer ---> |+>_L constructed.
    # We will measure syndromes in serial (with only one ancilla), to have fewer qubits in the simulation

    n = H_Z.shape[1]
    r = H_Z.shape[0]

    data_register = QuantumRegister(n, "data")
    ancilla_register = QuantumRegister(1, "anc")
    syn = ClassicalRegister(r, "syn")
    qc = QuantumCircuit(data_register, ancilla_register, syn)

    # 1. Prepare |+>^n
    for q in range(n):
        qc.h(data_register[q])

    # 2. Measure Z stabilizers
    for i in range(r):
        for q in range(n):
            if H_Z[i, q]:
                qc.cx(data_register[q], ancilla_register[0])

        # Measure and store in classical register
        qc.measure(ancilla_register[0], syn[i])
        qc.reset(ancilla_register[0])

    # 3. Correct syndrome
    lookup = make_syndrome_lookup(H_Z)
    for syndrome, correction_qubits in lookup.items():
        # No correction
        if syndrome == 0:
            continue
        # Correction
        with qc.if_test((syn, syndrome)):
            for q in correction_qubits:
                qc.x(data_register[q])

    return qc
