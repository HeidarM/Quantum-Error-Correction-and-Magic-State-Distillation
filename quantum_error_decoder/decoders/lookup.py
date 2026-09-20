# decoders/lookup.py
from itertools import combinations, product

import numpy as np

from qec.symplectic import symplectic_to_pauli


class LookupDecoder:
    def __init__(self, code):
        self.code = code
        self.table = self._build_table()

    # For each syndrome, assign one error out of potentially many possible.
    # Try errors in increasing weight and keep the first one found.
    def _build_table(self):
        table = {}

        n = self.code.n
        num_syndromes = 2 ** self.code.r
        
        # weight 0
        x = np.zeros(n, dtype=np.uint8)
        z = np.zeros(n, dtype=np.uint8)
        s = tuple(self.code.syndrome(x, z))
        table[s] = (x, z)

        # X = (1, 0), Y = (1, 1), Z = (0, 1).
        single_qubit_errors = ((1, 0), (1, 1), (0, 1))

        # Go through errors by weights 1, 2, 3, ...
        for weight in range(1, n + 1):
            
            # choose which qubits carry errors
            for qubits in combinations(range(n), weight):

                # choose X/Y/Z on those qubits, as (x, z) pairs
                for local_errors in product(single_qubit_errors, repeat=weight):
                    x = np.zeros(n, dtype=np.uint8)
                    z = np.zeros(n, dtype=np.uint8)

                    for q, (x_bit, z_bit) in zip(qubits, local_errors):
                        x[q] = x_bit
                        z[q] = z_bit

                    # Compute syndrome of error
                    s = tuple(self.code.syndrome(x, z))

                    # keep first representative found => minimum-weight representative (because we increase weight)
                    if s not in table:
                        table[s] = (x, z)

                    # all syndromes covered
                    if len(table) == num_syndromes:
                        return table

        return table

    def decode(self, syndrome):
        correction = self.table.get(tuple(syndrome))
        if correction is None:
            raise ValueError("No correction configured for this syndrome")

        x, z = correction
        return x.copy(), z.copy()

    def decode_to_pauli(self, syndrome):
        x, z = self.decode(syndrome)
        return symplectic_to_pauli(x, z)

    def print_table(self):
        title = "Binary decoder table"

        syndrome_bits = len(next(iter(self.table)))
        syndrome_width = max(len("syndrome"), syndrome_bits)
        correction_width = max(len("correction"), self.code.n)
        table_width = syndrome_width + 2 + correction_width

        print(title.center(table_width))
        print(f"{'syndrome':<{syndrome_width}}  {'correction':<{correction_width}}")
        print(f"{'-' * syndrome_width}  {'-' * correction_width}")

        for syndrome, (x, z) in sorted(self.table.items()):
            syndrome_text = "".join(str(bit) for bit in syndrome)
            correction = symplectic_to_pauli(x, z).replace("I", "-")
            print(f"{syndrome_text:<{syndrome_width}}  {correction}")
