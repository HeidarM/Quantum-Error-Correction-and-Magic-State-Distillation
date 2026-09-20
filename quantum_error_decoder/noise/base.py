# noise/base.py

# Interface for noise models

class NoiseModel:
    def sample(self, n, rng):
        raise NotImplementedError