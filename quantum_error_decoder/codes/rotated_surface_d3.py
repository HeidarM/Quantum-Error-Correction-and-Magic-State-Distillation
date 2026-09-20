# quantum_error_decoder/codes/rotated_surface_d3.py
# Rotated planar surface code [[9, 1, 3]]

from quantum_error_decoder.stabilizer_code import StabilizerCode

stabilizers = [
    "XXIXXIIII",
    "IXXIIIIII",
    "IIIIXXIXX",
    "IIIIIIXXI",

    "ZIIZIIIII",
    "IZZIZZIII",
    "IIIZZIZZI",
    "IIIIIZIIZ",
]

logical_x = "XIIXIIXII"   # left vertical string
logical_z = "ZZZIIIIII"   # top horizontal string

rotated_surface_d3 = StabilizerCode.from_pauli_strings(
    stabilizers,
    logical_x,
    logical_z,
)
