import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def save_trajectory(data, output_path):
    valid = data[data["detected"]]

    plt.figure(figsize=(8, 6))
    plt.plot(valid["x_px"], valid["y_px"], linewidth=1)
    plt.gca().invert_yaxis()
    plt.xlabel("X position (px)")
    plt.ylabel("Y position (px)")
    plt.title("Trajectory")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_heatmap(
    data,
    output_path,
    bins=60,
    cmap="viridis",
    vmin=None,
    vmax=None,
    normalize_percent=False,
    log_scale=False,
):
    valid = data[data["detected"]]

    if valid.empty:
        return

    weights = None

    if normalize_percent:
        weights = np.ones(len(valid), dtype=float) * (100.0 / len(valid))

    norm = None

    if log_scale:
        norm = LogNorm(
            vmin=max(vmin or 1e-6, 1e-6),
            vmax=vmax,
        )

    plt.figure(figsize=(8, 6))
    plt.hist2d(
        valid["x_px"],
        valid["y_px"],
        bins=bins,
        weights=weights,
        cmap=cmap,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        norm=norm,
    )
    plt.gca().invert_yaxis()
    plt.xlabel("X position (px)")
    plt.ylabel("Y position (px)")
    plt.title("Occupancy heatmap")
    plt.colorbar(
        label="Occupancy (%)" if normalize_percent else "Frames"
    )
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
