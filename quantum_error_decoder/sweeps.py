# quantum_error_decoder/sweeps.py

from functools import partial

import numpy as np

from qec.parallel import parallel_map
from quantum_error_decoder.monte_carlo import estimate_logical_failure_rate


def _estimate_at_p(index_and_p, code, decoder_factory, noise_factory, shots, seed):
    index, p = index_and_p
    decoder = decoder_factory(code)
    noise = noise_factory(p)
    p_logical = estimate_logical_failure_rate(
        code,
        decoder,
        noise,
        shots=shots,
        seed=seed + index,
    )

    return p, p_logical


# Estimate the logical-failure rate for every physical error probability in p_values.
def estimate_logical_failure_curve(
    code,
    decoder_factory,
    noise_factory,
    p_values,
    shots,
    num_workers=None,
    seed=7,
    progress_description="",
):
    worker = partial(
        _estimate_at_p,
        code=code,
        decoder_factory=decoder_factory,
        noise_factory=noise_factory,
        shots=shots,
        seed=seed,
    )
    results = parallel_map(
        worker,
        enumerate(p_values),
        num_workers=num_workers,
        progress_description=progress_description,
    )

    return np.array(results)
