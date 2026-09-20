# codes/shor9.py
# Shor's 9 qubit code

from quantum_error_decoder.stabilizer_code import StabilizerCode



stabilizers = [
    "ZZIIIIIII",
    "IZZIIIIII",
    "IIIZZIIII",
    "IIIIZZIII",
    "IIIIIIZZI",
    "IIIIIIIZZ",
    "XXXXXXIII",
    "IIIXXXXXX",
]

shor9 = StabilizerCode.from_pauli_strings(stabilizers,  logical_x="ZIIZIIZII", logical_z="XXXIIIIII",)
