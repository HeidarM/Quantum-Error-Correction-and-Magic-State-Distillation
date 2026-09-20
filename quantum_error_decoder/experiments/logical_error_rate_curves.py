# quantum_error_decoder/experiments/logical_error_rate_curves.py
# Run from the repository root as: python -m quantum_error_decoder.experiments.logical_error_rate_curves

from pathlib import Path

import numpy as np

from quantum_error_decoder.codes.planar_surface import planar_surface_code
from quantum_error_decoder.codes.rotated_surface_d3 import rotated_surface_d3
from quantum_error_decoder.codes.shor9 import shor9
from quantum_error_decoder.decoders.lookup import LookupDecoder
from quantum_error_decoder.noise.depolarizing import DepolarizingNoise
from quantum_error_decoder.plots import plot_logical_error_rates
from quantum_error_decoder.sweeps import estimate_logical_failure_curve


if __name__ == "__main__":
    planar_surface_d3 = planar_surface_code(3)

    codes_to_compare = [
        (shor9, "Shor-9"),
        (rotated_surface_d3, "Rotated surface-d3"),
        (planar_surface_d3, "Planar surface-d3"),
    ]

    p_values = np.arange(0.001, 0.15, 0.002)
    shots = 40000

    curves = []
    for code, label in codes_to_compare:
        results = estimate_logical_failure_curve(
            code,
            LookupDecoder,
            DepolarizingNoise,
            p_values,
            shots,
            progress_description=label,
        )
        curves.append((results, label))

    output_path = Path(__file__).resolve().parents[2] / "figures" / "logical_error_rate_curves.png"
    plot_logical_error_rates(curves, log_scale=False, save_path=output_path)
    print(f"Saved figure: {output_path}")
