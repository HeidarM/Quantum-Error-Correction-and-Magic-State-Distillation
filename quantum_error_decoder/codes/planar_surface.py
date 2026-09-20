# quantum_error_decoder/codes/planar_surface.py
# General unrotated planar surface code with parameters
# [[d^2 + (d - 1)^2, 1, d]].

import numpy as np

from quantum_error_decoder.stabilizer_code import StabilizerCode

# Lattice geometry:
#
# Vertices are (x, y), with x = 0, ..., d - 1 and y = 0, ..., d.
# There are d^2 vertical edges v(x, y), from (x, y) to (x, y + 1),
# and (d - 1)^2 horizontal edges h(x, y), from (x, y) to (x + 1, y),
# with y = 1, ..., d - 1. Thus n = d^2 + (d - 1)^2 data qubits.
#
# The top and bottom edges, y = 0 and y = d, are rough boundaries:
# there are no X stars there, so direct Z strings may end there and
# their e anyons are condensed. The left and right sides, x = 0 and
# x = d - 1, are smooth boundaries: dual X strings may end there and
# their m fluxes are condensed.
#
# Logical strings are nontrivial relative homology classes: they cannot be
# deformed into stabilizers while keeping their endpoints on the allowed boundaries.
def planar_surface_code(distance):
    d = distance

    # vertices:       (x, y)
    # edges:          ("v", x, y) and ("h", x, y)
    # line operators: binary edge-support vectors; 1 selects an edge
    vertical_edges = [
        ("v", x, y)
        for x in range(d)
        for y in range(d)
    ]
    horizontal_edges = [
        ("h", x, y)
        for x in range(d - 1)
        for y in range(1, d)
    ]
    data_edges = vertical_edges + horizontal_edges
    edge_index = {edge: index for index, edge in enumerate(data_edges)}
    n = len(data_edges)

    # For each allowed vertex v, construct the star
    # S_X(v) = product_{e incident on v} X_e.
    # These are weight four in the bulk and weight three at smooth boundaries.
    # There are no stars on rough boundaries, so a Z string may end there without an e excitation.
    # H_X[vertex, edge] = 1 iff edge is incident on that vertex.
    x_check_rows = []
    x_check_vertices = []
    for x in range(d):
        for y in range(1, d):
            row = np.zeros(n, dtype=np.uint8)
            incident_edges = [("v", x, y - 1), ("v", x, y)]

            if x > 0:
                incident_edges.append(("h", x - 1, y))
            if x < d - 1:
                incident_edges.append(("h", x, y))

            for edge in incident_edges:
                row[edge_index[edge]] = 1

            x_check_rows.append(row)
            x_check_vertices.append((x, y))

    # For each face f, construct the plaquette check
    # S_Z(f) = product_{e on boundary of f} Z_e.
    # These measure m flux, the endpoints of dual X strings. A face next to a rough boundary has only three retained edges.
    # H_Z[face, edge] = 1 iff edge lies on that face boundary.
    z_check_rows = []
    z_check_faces = []
    for x in range(d - 1):
        for y in range(d):
            row = np.zeros(n, dtype=np.uint8)
            boundary_edges = [("v", x, y), ("v", x + 1, y)]

            if y > 0:
                boundary_edges.append(("h", x, y))
            if y < d - 1:
                boundary_edges.append(("h", x, y + 1))

            for edge in boundary_edges:
                row[edge_index[edge]] = 1

            z_check_rows.append(row)
            z_check_faces.append((x, y))

    h_x = np.array(x_check_rows, dtype=np.uint8)
    h_z = np.array(z_check_rows, dtype=np.uint8)
    zero_x_checks = np.zeros_like(h_z)
    zero_z_checks = np.zeros_like(h_x)

    # ---- Relative homology representatives ----
    # Logical Z: vertical rough-to-rough edge string.
    logical_z_support = np.zeros(n, dtype=np.uint8)
    for y in range(d):
        logical_z_support[edge_index[("v", 0, y)]] = 1

    # Logical X: horizontal smooth-to-smooth dual string.
    logical_x_support = np.zeros(n, dtype=np.uint8)
    for x in range(d):
        logical_x_support[edge_index[("v", x, 0)]] = 1

    logical_x = "".join("X" if bit else "I" for bit in logical_x_support)
    logical_z = "".join("Z" if bit else "I" for bit in logical_z_support)

    code = StabilizerCode(
        np.vstack([h_x, zero_x_checks]),
        np.vstack([zero_z_checks, h_z]),
        logical_x=logical_x,
        logical_z=logical_z,
    )

    # Retain the edge geometry so strings and syndromes can be inspected.
    code.data_edges = data_edges
    code.edge_index = edge_index
    code.h_x = h_x
    code.h_z = h_z
    code.x_check_vertices = x_check_vertices
    code.z_check_faces = z_check_faces
    code.distance = d

    return code
