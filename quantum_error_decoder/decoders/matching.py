# quantum_error_decoder/decoders/matching.py

import numpy as np
import pymatching


# Minimum-weight perfect matching decoder for CSS codes with graphlike checks.
class MatchingDecoder:
    def __init__(self, code):
        self.code = code

        has_x = np.any(code.Mx, axis=1)
        has_z = np.any(code.Mz, axis=1)

        # CSS stabilizers are either X-only or Z-only.
        if np.any(has_x & has_z) or np.any(~has_x & ~has_z):
            raise ValueError("MatchingDecoder requires a CSS code with non-identity checks")

        self.x_check_rows = np.flatnonzero(has_x)
        self.z_check_rows = np.flatnonzero(has_z)

        # A Z error creates defects at X checks, and an X error creates defects at Z checks.
        self.z_error_matching = pymatching.Matching(code.Mx[self.x_check_rows])
        self.x_error_matching = pymatching.Matching(code.Mz[self.z_check_rows])

    def decode(self, syndrome):
        syndrome = np.asarray(syndrome, dtype=np.uint8)

        # Matching pairs syndrome defects, or connects a defect to a boundary,
        # with a minimum-weight correction path.
        correction_z = self.z_error_matching.decode(syndrome[self.x_check_rows])
        correction_x = self.x_error_matching.decode(syndrome[self.z_check_rows])

        return correction_x.astype(np.uint8), correction_z.astype(np.uint8)
