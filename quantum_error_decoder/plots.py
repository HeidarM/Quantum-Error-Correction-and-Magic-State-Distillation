# quantum_error_decoder/plots.py

from pathlib import Path

import numpy as np


def save_figure(fig, save_path):
    """Save a figure, creating its output directory when needed."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")


def plot_logical_error_rate(
    results,
    label=None,
    log_scale=False,
    ax=None,
    show_break_even=True,
    save_path=None,
    show=True,
    title="Logical error rate of a QEC code",
):
    return plot_logical_error_rates(
        [(results, label)],
        log_scale=log_scale,
        ax=ax,
        show_break_even=show_break_even,
        save_path=save_path,
        show=show,
        title=title,
    )


def plot_logical_error_rates(
    curves,
    log_scale=False,
    ax=None,
    show_break_even=True,
    save_path=None,
    show=True,
    title="Logical error rate of a QEC code",
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
    else:
        fig = ax.figure

    for results, label in curves:
        p = np.asarray(results[:, 0])
        p_logical = np.asarray(results[:, 1])

        idx = np.argsort(p)
        ax.plot(
            p[idx],
            p_logical[idx],
            marker="o",
            markersize=5,
            linewidth=2.2,
            label=label or r"Logical error rate $p_L$",
            zorder=3,
        )

    if show_break_even:
        ax.plot(
            p,
            p,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=r"break-even: $p_L = p$",
            zorder=2,
        )

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel(r"Physical error rate, $p$", fontsize=12)
    ax.set_ylabel(r"Logical error rate, $p_L$", fontsize=12)

    ax.set_title(
        title,
        fontsize=14,
        pad=10,
    )

    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=0.6,
        alpha=0.25,
    )
    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.5,
        alpha=0.15,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        direction="out",
        length=5,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=3,
    )

    ax.legend(
        frameon=False,
        fontsize=10,
        loc="best",
    )

    fig.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)

    if show:
        plt.show()

    return fig, ax


def _plot_planar_lattice(ax, code):
    for edge in code.data_edges:
        kind, x, y = edge
        if kind == "v":
            ax.plot([x, x], [y, y + 1], color="0.86", linewidth=0.9, zorder=1)
        else:
            ax.plot([x, x + 1], [y, y], color="0.86", linewidth=0.9, zorder=1)

    # Z strings may terminate on the horizontal rough boundaries; dual X
    # strings may terminate on the vertical smooth boundaries.
    for y in (0, code.distance):
        ax.plot([-0.35, code.distance - 0.65], [y, y],
                color="#555555", linewidth=3.2, zorder=2,
                solid_capstyle="butt")
    for x in (-0.35, code.distance - 0.65):
        ax.plot([x, x], [0.25, code.distance - 0.25],
                color="#555555", linewidth=3.2, linestyle=(0, (2, 2)),
                zorder=2, dash_capstyle="butt")

    ax.set_aspect("equal")
    ax.set_xlim(-0.8, code.distance - 0.2)
    ax.set_ylim(-0.4, code.distance + 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_planar_edge_support(
    ax, code, support, color, zorder=3, linestyle="-", dual=False,
):
    for edge_index in np.flatnonzero(support):
        kind, x, y = code.data_edges[edge_index]
        if dual and kind == "v":
            # The dual edge crosses the physical vertical data edge.
            xs, ys = [x - 0.5, x + 0.5], [y + 0.5, y + 0.5]
        elif dual:
            # The dual edge crosses the physical horizontal data edge.
            xs, ys = [x + 0.5, x + 0.5], [y - 0.5, y + 0.5]
        elif kind == "v":
            xs, ys = [x, x], [y, y + 1]
        else:
            xs, ys = [x, x + 1], [y, y]
        ax.plot(xs, ys, color=color, linewidth=3.5, linestyle=linestyle,
                solid_capstyle="round", solid_joinstyle="round", zorder=zorder)


def _edge_midpoints(code, support):
    points = []
    for edge_index in np.flatnonzero(support):
        kind, x, y = code.data_edges[edge_index]
        points.append((x, y + 0.5) if kind == "v" else (x + 0.5, y))
    return points


def _plot_pauli_support(ax, code, x, z, linestyle="-"):
    """Draw e/Z strings on the direct lattice and m/X strings on its dual."""
    x = np.asarray(x, dtype=np.uint8)
    z = np.asarray(z, dtype=np.uint8)
    y_support = x & z

    # A Y operator has both components, so it appears as crossing direct and
    # dual segments. The diamond marks the physical data qubit at the crossing.
    _plot_planar_edge_support(ax, code, z, "#D55E00",
                              linestyle=linestyle)
    _plot_planar_edge_support(ax, code, x, "#0072B2",
                              linestyle=linestyle, dual=True)
    y_points = _edge_midpoints(code, y_support)
    if y_points:
        xs, ys = zip(*y_points)
        ax.scatter(xs, ys, s=28, marker="D", color="#CC79A7",
                   edgecolor="white", linewidth=0.7, zorder=6)


def plot_planar_matching_examples(code, examples, show=True, save_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not examples:
        raise ValueError("At least one matching example is required")

    x_check_rows = np.flatnonzero(np.any(code.Mx, axis=1))
    z_check_rows = np.flatnonzero(np.any(code.Mz, axis=1))
    fig, axes = plt.subplots(
        len(examples), 3, figsize=(10, 3.5 * len(examples) + 1.1),
        sharex=True, sharey=True, squeeze=False,
    )
    success_color, failure_color = "#009E73", "#AA3377"

    for row, (label, error_x, error_z, correction_x, correction_z, syndrome, success) in enumerate(examples):
        error_x = np.asarray(error_x, dtype=np.uint8)
        error_z = np.asarray(error_z, dtype=np.uint8)
        correction_x = np.asarray(correction_x, dtype=np.uint8)
        correction_z = np.asarray(correction_z, dtype=np.uint8)
        residual_x = error_x ^ correction_x
        residual_z = error_z ^ correction_z
        x_defect_vertices = [
            vertex
            for vertex, defect in zip(code.x_check_vertices, syndrome[x_check_rows])
            if defect
        ]
        z_defect_faces = [
            (x + 0.5, y + 0.5)
            for (x, y), defect in zip(code.z_check_faces, syndrome[z_check_rows])
            if defect
        ]

        error_axis, correction_axis, residual_axis = axes[row]
        residual_color = success_color if success else failure_color
        panels = (
            (error_axis, error_x, error_z, "-"),
            # The column heading already distinguishes corrections. Keeping
            # paths solid prevents dash resets from opening gaps at corners.
            (correction_axis, correction_x, correction_z, "-"),
            (residual_axis, residual_x, residual_z, "-"),
        )
        for ax, x_support, z_support, style in panels:
            _plot_planar_lattice(ax, code)
            _plot_pauli_support(ax, code, x_support, z_support,
                                linestyle=style)

        # Show the same observed defects beside the error and its correction.
        for ax in (error_axis, correction_axis):
            for points, marker in ((x_defect_vertices, "o"), (z_defect_faces, "s")):
                if points:
                    xs, ys = zip(*points)
                    ax.scatter(xs, ys, s=34, marker=marker, color="#222222",
                               edgecolor="white", linewidth=0.8, zorder=5)

        error_axis.set_ylabel(label, fontsize=11, labelpad=12)
        if success:
            description = "Stabilizer loop: success" if (residual_x.any() or residual_z.any()) else "No residual: success"
        else:
            description = "Logical operator: failure"
        residual_axis.set_xlabel(description, color=residual_color, fontsize=11, labelpad=8)

    for ax, title in zip(axes[0], ("Physical error", "Matching correction", "Residual after correction")):
        ax.set_title(title, fontsize=12, pad=12)

    legend = [
        Line2D([], [], color="#0072B2", linewidth=3,
               label=r"$m$ string ($X$, dual)"),
        Line2D([], [], color="#D55E00", linewidth=3,
               label=r"$e$ string ($Z$, direct)"),
        Line2D([], [], color="none", marker="D", markerfacecolor="#CC79A7",
               markeredgecolor="white", markersize=6, label=r"$Y=XZ$"),
        Line2D([], [], color="none", marker="o", markerfacecolor="#222222",
               markersize=5.5, label=r"$X$-check defect"),
        Line2D([], [], color="none", marker="s", markerfacecolor="#222222",
               markersize=5.5, label=r"$Z$-check defect"),
        Line2D([], [], color="#555555", linewidth=3, label="Rough boundary"),
        Line2D([], [], color="#555555", linewidth=3, linestyle=(0, (2, 2)),
               label="Smooth boundary"),
    ]
    fig.suptitle("Surface-code matching: success and logical failure", fontsize=16, y=0.985)
    fig.legend(handles=legend, loc="lower center", ncol=4,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.015))
    # Leave explicit room for each row's outcome label and the shared legend.
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.86,
                        hspace=0.38, wspace=0.18)

    if save_path is not None:
        save_figure(fig, save_path)

    if show:
        plt.show()

    return fig, axes
