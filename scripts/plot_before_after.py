#!/usr/bin/env python3
"""Before/after scatter of one synthetic cloud: raw input next to the filtered output.

Reads the raw float32 xyz files written by `bench_filter --dump PREFIX` and writes a PNG.

    bench_filter --backend cuda --res 320x240 --iters 1 --warmup 0 --dump /tmp/cloud
    python3 scripts/plot_before_after.py /tmp/cloud_320x240_input.xyz /tmp/cloud_320x240_cuda.xyz \
        results/before_after.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MIN_RANGE, MAX_RANGE = 0.3, 10.0  # the filter's defaults, for the caption only


def load(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 3)


def main(raw_path, out_path, png_path):
    raw, out = load(raw_path), load(out_path)
    finite = raw[np.isfinite(raw).all(axis=1)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    titles = [f"Input: {len(raw):,} points ({len(raw) - len(finite):,} NaN not shown)",
              f"Output: {len(out):,} voxel centroids (0.05 m voxels, range {MIN_RANGE}-{MAX_RANGE} m)"]
    for col, pts in enumerate((finite, out)):
        # Row 0: as the camera sees it (x across, height = -y up), colour = depth z.
        ax = axes[0, col]
        sc0 = ax.scatter(pts[:, 0], -pts[:, 1], c=pts[:, 2], s=0.4, cmap="viridis", vmin=0, vmax=12,
                         linewidths=0, rasterized=True)
        ax.set_title(titles[col], fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("height, -y (m)")
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-2, 4.5)
        # Row 1: top-down (x across, depth z up), colour = height; shows the range cut.
        ax = axes[1, col]
        sc1 = ax.scatter(pts[:, 0], pts[:, 2], c=-pts[:, 1], s=0.4, cmap="plasma", vmin=-1.5, vmax=1.5,
                         linewidths=0, rasterized=True)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z, depth (m)")
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(0, 12)
        ax.set_aspect("equal")
    fig.colorbar(sc0, ax=axes[0, :], label="depth z (m)", shrink=0.85)
    fig.colorbar(sc1, ax=axes[1, :], label="height -y (m)", shrink=0.85)
    fig.suptitle("Synthetic 320x240 depth-camera cloud: camera view (top), top-down view (bottom)", fontsize=11)
    fig.savefig(png_path, dpi=110)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
