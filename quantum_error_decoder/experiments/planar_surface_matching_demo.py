# quantum_error_decoder/experiments/planar_surface_matching_demo.py
# Run from the repository root as: python -m quantum_error_decoder.experiments.planar_surface_matching_demo

from pathlib import Path

import numpy as np

from quantum_error_decoder.codes.planar_surface import planar_surface_code
from quantum_error_decoder.decoders.matching import MatchingDecoder
from quantum_error_decoder.plots import plot_planar_matching_examples


if __name__ == "__main__":
    code = planar_surface_code(13)
    decoder = MatchingDecoder(code)

    # Each pair contains Z support followed by X support. An edge appearing
    # in both lists carries a Y error.
    error_string_examples = [
        (
            "Mixed Pauli errors",
            # Two Z strings in the left half, plus the Z component of the
            # isolated central Y error.
            [("v", 2, 2), ("h", 2, 3), ("v", 3, 2),
             ("h", 2, 9), ("h", 3, 9), ("v", 4, 9),
             ("v", 6, 6)],
            # Two dual X strings in the right half, plus the X component of
            # the same central Y error. The groups are deliberately spaced
            # apart so each error and its defects remain legible.
            [("v", 9, 2), ("v", 10, 2), ("h", 10, 3),
             ("h", 8, 9), ("h", 8, 10), ("v", 9, 10),
             ("v", 10, 10),
             ("v", 6, 6)],
        ),
        (
            "Logical failure",
            # This two-ended Z string is longer than half the code distance.
            # Its endpoints are near opposite rough boundaries, so matching
            # closes each end to the nearest boundary. The residual is then a
            # nontrivial rough-to-rough logical Z string.
            [("v", 7, y) for y in range(2, 11)],
            [],
        ),
    ]

    examples = []
    for label, z_error_string, x_error_string in error_string_examples:
        error_x = np.zeros(code.n, dtype=np.uint8)
        error_z = np.zeros(code.n, dtype=np.uint8)
        for edge in z_error_string:
            error_z[code.edge_index[edge]] = 1
        for edge in x_error_string:
            error_x[code.edge_index[edge]] = 1

        syndrome = code.syndrome(error_x, error_z)
        correction_x, correction_z = decoder.decode(syndrome)
        success = code.is_stabilizer(error_x ^ correction_x, error_z ^ correction_z)
        examples.append((label, error_x, error_z, correction_x, correction_z, syndrome, success))
        print(f"{label}: matching decoder success = {success}")

    output_path = Path(__file__).resolve().parents[2] / "figures" / "planar_surface_matching.png"
    plot_planar_matching_examples(code, examples, save_path=output_path)
    print(f"Saved figure: {output_path}")
