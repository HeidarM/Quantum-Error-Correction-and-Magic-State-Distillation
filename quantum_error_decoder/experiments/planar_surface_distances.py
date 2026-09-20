# quantum_error_decoder/experiments/planar_surface_distances.py
# Run from the repository root as: python -m quantum_error_decoder.experiments.planar_surface_distances

from pathlib import Path

import numpy as np

from quantum_error_decoder.codes.planar_surface import planar_surface_code
from quantum_error_decoder.decoders.matching import MatchingDecoder
from quantum_error_decoder.noise.depolarizing import DepolarizingNoise
from quantum_error_decoder.plots import plot_logical_error_rates
from quantum_error_decoder.sweeps import estimate_logical_failure_curve


if __name__ == "__main__":
    distances = [3, 5, 7, 9, 13]
    codes_to_plot = [
        (planar_surface_code(d), f"Planar surface-d{d}")
        for d in distances
    ]

    p_values = np.arange(0.01, 0.2, 0.01)
    shots = 40000
    curves = []

    for code, label in codes_to_plot:
        results = estimate_logical_failure_curve(
            code,
            MatchingDecoder,
            DepolarizingNoise,
            p_values,
            shots,
            progress_description=label,
        )
        curves.append((results, label))

    output_path = Path(__file__).resolve().parents[2] / "figures" / "planar_surface_distances.png"
    plot_logical_error_rates(
        curves, log_scale=False, save_path=output_path,
        title="Surface-code decoding across code distances",
    )
    print(f"Saved figure: {output_path}")
