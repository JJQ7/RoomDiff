"""Top-down and side-view PNGs of a reconstruction, to judge the before/after alignment.

Usage: python scripts/plot_align.py <out_dir> [--png-dir DIR]

Reads `<out>/recon/{before,after}.ply`, the cameras in `recon.npz` and `meta.json`, and
writes `<run>_top.png` (all points; a +-5 cm horizontal slice at camera height; after
points coloured by their distance to the before cloud) and `<run>_side.png` (two side
views) to `--png-dir` (default `<out>/recon`), where `<run>` is the name of `<out>`.
Blue = before, orange = after. The clouds are levelled with a gravity estimate from the
cameras; at most 300k random points per cloud are drawn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from plyfile import PlyData
from scipy.spatial import cKDTree

RECORDINGS = ("before", "after")
COLORS = {"before": "#2a78d6", "after": "#eb6834"}
DIST_CMAP = LinearSegmentedColormap.from_list(
    "dist", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
MAX_POINTS = 300_000
MARGIN = 0.3  # m around the 1st-99th percentile box


def level_rotation(cam2world: np.ndarray) -> np.ndarray:
    """Rotation that maps the estimated gravity ("down") direction onto +y.

    A phone held upright keeps its x-axis horizontal, so "down" is the direction most
    orthogonal to all camera x-axes; its sign follows the mean camera y-axis (OpenCV y
    points down).
    """
    down = np.linalg.svd(cam2world[:, :3, 0].astype(np.float64))[2][-1]
    down *= np.sign(down @ cam2world[:, :3, 1].mean(0))
    down /= np.linalg.norm(down)
    v, c = np.cross(down, [0.0, 1.0, 0.0]), float(down[1])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


class AlignPlot:
    """Levelled clouds and camera paths of one run, plus the plotting helpers."""

    def __init__(self, out: Path) -> None:
        recon_dir = out / "recon"
        with np.load(recon_dir / "recon.npz") as npz:  # lazy: only the cameras are read
            cams, rec = npz["cam2world"], npz["rec"]
        meta = json.loads((recon_dir / "meta.json").read_text())
        R = level_rotation(cams)
        rng = np.random.default_rng(0)
        self.full, self.clouds, self.paths = {}, {}, {}
        for r, name in enumerate(RECORDINGS):
            v = PlyData.read(str(recon_dir / f"{name}.ply"))["vertex"]
            pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64) @ R.T
            self.full[name] = pts
            if len(pts) > MAX_POINTS:
                pts = pts[rng.choice(len(pts), MAX_POINTS, replace=False)]
            self.clouds[name] = pts
            self.paths[name] = cams[rec == r, :3, 3] @ R.T
        centres = np.concatenate(list(self.paths.values()))
        lo, hi = np.percentile(np.concatenate(list(self.clouds.values())), [1, 99], axis=0)
        self.lo, self.hi = np.minimum(lo, centres.min(0)), np.maximum(hi, centres.max(0))
        self.camera_height = float(np.median(centres[:, 1]))
        d = meta["diagnostics"]
        covis = d["nn_covisible"]
        medians = [f"{100 * covis[k]['median_m']:.1f} cm" if k in covis else "n/a"
                   for k in ("before_to_after", "after_to_before")]
        self.title = (f"{out.name} ({meta['mode']}): {d['verdict']}; co-visible median "
                      f"b->a {medians[0]}, a->b {medians[1]}")

    def _frame(self, ax: Axes, i: int, j: int, flip: bool) -> None:
        jl, jh = (-self.hi[j], -self.lo[j]) if flip else (self.lo[j], self.hi[j])
        ax.set_xlim(self.lo[i] - MARGIN, self.hi[i] + MARGIN)
        ax.set_ylim(jl - MARGIN, jh + MARGIN)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3, lw=0.5)

    def cameras(self, ax: Axes, i: int, j: int, flip: bool = False) -> None:
        """Camera paths (dots), first camera marked with a black triangle."""
        sign = -1 if flip else 1
        for name, c in self.paths.items():
            ax.plot(c[:, i], sign * c[:, j], "-o", ms=3, lw=1.5, color=COLORS[name],
                    label=f"{name} cams ({len(c)})")
            ax.plot(c[0, i], sign * c[0, j], "k^", ms=6)
        self._frame(ax, i, j, flip)

    def scatter(self, ax: Axes, i: int, j: int, flip: bool = False,
                slab: float | None = None, size: float = 0.15) -> None:
        """Both clouds projected on axes (i, j); `slab`: keep |y - camera height| < slab."""
        sign = -1 if flip else 1
        for name, pts in self.clouds.items():
            if slab is not None:
                pts = pts[np.abs(pts[:, 1] - self.camera_height) < slab]
            ax.scatter(pts[:, i], sign * pts[:, j], s=size, alpha=0.25, color=COLORS[name],
                       linewidths=0, rasterized=True)
        self.cameras(ax, i, j, flip)


def plot(out: Path, png_dir: Path) -> tuple[Path, Path]:
    """Write `<run>_top.png` and `<run>_side.png` to `png_dir`; return their paths."""
    p = AlignPlot(out)
    png_dir.mkdir(parents=True, exist_ok=True)
    top, side = png_dir / f"{out.name}_top.png", png_dir / f"{out.name}_side.png"

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), layout="constrained")
    p.scatter(axes[0], 0, 2)
    axes[0].set(title="top-down, all points (blue before, orange after; ^ = first cam)",
                xlabel="x (m)", ylabel="z (m)")
    axes[0].legend(loc="upper right")
    p.scatter(axes[1], 0, 2, slab=0.05, size=0.8)
    axes[1].set(title="horizontal slice +-5 cm at camera height", xlabel="x (m)",
                ylabel="z (m)")
    dist = cKDTree(p.full["before"]).query(p.clouds["after"], workers=8)[0]
    order = np.argsort(dist)  # draw far points last so they stay visible
    after = p.clouds["after"][order]
    sc = axes[2].scatter(after[:, 0], after[:, 2], c=np.clip(dist[order], 0, 0.3), s=0.3,
                         cmap=DIST_CMAP, vmin=0, vmax=0.3, linewidths=0, rasterized=True)
    p.cameras(axes[2], 0, 2)
    fig.colorbar(sc, ax=axes[2], shrink=0.7,
                 label="distance to nearest before point (m, clipped at 0.3)")
    axes[2].set(title="after points by distance to the before cloud", xlabel="x (m)",
                ylabel="z (m)")
    fig.suptitle(p.title)
    fig.savefig(top, dpi=90)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), layout="constrained")
    p.scatter(axes[0], 0, 1, flip=True)
    axes[0].set(title="side view (looking along z)", xlabel="x (m)", ylabel="up (m)")
    axes[0].legend(loc="upper right")
    p.scatter(axes[1], 2, 1, flip=True)
    axes[1].set(title="side view (looking along x)", xlabel="z (m)", ylabel="up (m)")
    fig.suptitle(p.title)
    fig.savefig(side, dpi=90)
    plt.close(fig)
    return top, side


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path, help="run directory containing recon/")
    parser.add_argument("--png-dir", type=Path, help="output folder (default <out>/recon)")
    args = parser.parse_args()
    top, side = plot(args.out_dir, args.png_dir or args.out_dir / "recon")
    print(f"wrote {top} and {side}")


if __name__ == "__main__":
    main()
