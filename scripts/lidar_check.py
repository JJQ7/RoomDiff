"""Metric check of a RoomDiff run against iPhone LiDAR scans of the room.

Usage: python scripts/lidar_check.py <out_dir> --before lidar/before.ply
                                     --after lidar/after.ply [--json]

Scans: "3D Scanner App" exports, PLY point clouds or OBJ meshes (their vertices), in the
ARKit world frame (metres, gravity-aligned, +Y up, arbitrary yaw and origin; `--lidar-up z`
for a Z-up export).
Ours: `<out>/viewer/<rec>.ply`, else `<out>/recon/<rec>.ply`, else the valid points of
that recording in `<out>/recon/recon.npz` (levelled frame, +Y up, approximately metric).

Per recording, both clouds are voxel-downsampled to 2 cm and the similarity
x_lidar = s R x_ours + t is estimated from the points 0.1-2 m above each cloud's floor
(walls and furniture; floor and ceiling fit almost any horizontal pose). Both frames are
+Y up, so a coarse grid covers the scale s (0.60-1.40, step 0.025) and the yaw about +Y
(0-350 deg, step 10). Per candidate the vertical offset comes from the floor heights and
the horizontal one from the cross-correlation of top-down occupancy maps (5 cm cells),
which, unlike the centroids, still works when the scan covers more or less of the room
than the video. Score = geometric mean of the fraction of our points within 5 cm of the
scan, counted only over our points that land inside the scan's box (a scan cropped to
part of the room, e.g. one table, must not punish the rest of ours; the height band also
stops where the scan stops), and of the scan's points within 5 cm of ours (one direction
alone favours shrinking our cloud into a scan that covers part of it). The three best
distinct candidates get a
local pattern search (steps from 2.5 % / 5 deg down to 0.3 % / 0.6 deg, scored at 3 cm)
and a similarity ICP (Umeyama on mutual nearest neighbours within 8 -> 3 cm, so points
beyond the edge of the other cloud do not bias the scale; the rotation is free, so a
levelling error shows up as `tilt`); the best one at 3 cm wins.

Reports per recording s (ours -> metres), yaw, tilt, the residual median / p90 (m) of our
points within 10 cm of the scan and that overlap fraction; the before / after scale
consistency (two scans of one room should give the same s within a few percent); and the
changes of changes.json with their distance (moved: `translation_m`) and `obb` size times
s (the geometric mean of the fits), to compare with tape measurements. `--json` also
writes `<out>/lidar_check.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # roomdiff without installing
from roomdiff.geometry import rotation_y, sim3_matrix, umeyama, voxel_downsample
from roomdiff.paths import RECORDINGS
from roomdiff.recon import valid_points

VOXEL = 0.02  # m, both clouds
SCALES = np.round(np.arange(0.60, 1.40 + 1e-9, 0.025), 4)
YAWS = np.arange(0.0, 360.0, 10.0)
CELL = 0.05  # m, top-down occupancy cells
BAND = (0.1, 2.0)  # m above the floor: walls and furniture, no floor and no ceiling
INLIER = 0.05  # m, coarse score: points this close count (tolerates the 10 deg yaw steps)
FINE = 0.03  # m, the same for the local search and the final choice: sharper in scale
OVERLAP = 0.10  # m, residual statistics over our points this close to the scan
N_BAND, N_SCORE, N_ICP = 30_000, 2_000, 30_000  # point budgets (random subsets)
MIN_INSIDE = 200  # of the N_SCORE forward points must land inside the scan's box
TOP = 3  # coarse candidates refined
ICP_STEPS = (0.08, 0.05, 0.03)  # m, correspondence distances
ICP_ITERATIONS = 20  # per distance
CONSISTENT = 0.03  # before / after scales should agree within this fraction


def read_obj_vertices(path: Path) -> np.ndarray:
    """Vertex positions (N, 3) float64 of a Wavefront OBJ (`v x y z` lines; `vt`, `vn`,
    faces, comments and material lines are skipped)."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    rows.append(parts[1:4])
    if not rows:
        raise ValueError(f"{path}: no vertices")
    return np.asarray(rows, np.float64)


def read_ply(path: Path, up: str = "y") -> np.ndarray:
    """Vertex positions (N, 3) float64 of a PLY or OBJ file, converted to +Y up."""
    if Path(path).suffix.lower() == ".obj":
        points = read_obj_vertices(Path(path))
    else:
        vertex = PlyData.read(str(path))["vertex"]
        points = np.column_stack([np.asarray(vertex[a], np.float64) for a in "xyz"])
    points = points[np.isfinite(points).all(1)]
    return points[:, [0, 2, 1]] * [1.0, 1.0, -1.0] if up == "z" else points


def our_cloud(out: Path, rec: str) -> tuple[np.ndarray, str]:
    """Our points of one recording and where they came from."""
    for path in (out / "viewer" / f"{rec}.ply", out / "recon" / f"{rec}.ply"):
        if path.is_file():
            return read_ply(path), str(path)
    npz = out / "recon" / "recon.npz"
    if not npz.is_file():
        raise FileNotFoundError(f"no viewer/{rec}.ply, recon/{rec}.ply or recon.npz in {out}")
    with np.load(npz) as f:  # only the arrays needed for the points
        recon = {k: f[k] for k in ("rec", "valid", "depth", "K", "cam2world")}
    points, _ = valid_points(recon, np.flatnonzero(recon["rec"] == RECORDINGS.index(rec)))
    return points.astype(np.float64), f"{npz} ({rec} keyframes)"


def downsample(points: np.ndarray) -> np.ndarray:
    return voxel_downsample(points, None, VOXEL)[0].astype(np.float64)


def subset(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(points) <= n:
        return points
    return points[np.random.default_rng(seed).choice(len(points), n, replace=False)]


def floor_height(y: np.ndarray) -> float:
    """Lowest height layer (1 cm bins, smoothed over 5 cm) within 0.6 m above the 1st
    percentile that is a local peak with at least 30 % of the densest layer's points: a
    bed or a sofa seat can hold more points than a half-hidden floor."""
    lo = float(np.percentile(y, 1.0))
    hist, edges = np.histogram(y, bins=np.arange(lo - 0.1, lo + 0.6, 0.01))
    smooth = np.convolve(hist, np.ones(5), mode="same")
    peaks = [k for k in range(1, len(smooth) - 1) if smooth[k] >= 0.3 * smooth.max()
             and smooth[k] >= smooth[k - 1] and smooth[k] >= smooth[k + 1]]
    k = peaks[0] if peaks else int(np.argmax(smooth))
    return float(edges[k] + 0.005)


def occupancy(xz: np.ndarray, origin: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """log(1 + points per cell) of a top-down map."""
    ij = np.floor((xz - origin) / CELL).astype(np.int64)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    counts = np.bincount(ij[ok, 0] * shape[1] + ij[ok, 1], minlength=shape[0] * shape[1])
    return np.log1p(counts.reshape(shape).astype(np.float64))


def similarity(s: float, yaw: float, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = s * rotation_y(yaw), t
    return M


def inlier_fraction(M: np.ndarray, points: np.ndarray, tree: cKDTree, dist: float) -> float:
    d, _ = tree.query(points @ M[:3, :3].T + M[:3, 3], distance_upper_bound=dist, workers=-1)
    return float(np.isfinite(d).mean())


class Search:
    """Translation and score of a (scale, yaw) candidate (module docstring)."""

    def __init__(self, ours: np.ndarray, lidar: np.ndarray) -> None:
        self.ours = ours
        self.floor_o, self.floor_l = floor_height(ours[:, 1]), floor_height(lidar[:, 1])
        h = lidar[:, 1] - self.floor_l
        self.top = min(BAND[1], float(np.percentile(h, 99.5)) + 0.05)  # a cropped scan
        self.lidar = lidar[(h > BAND[0]) & (h < self.top)]
        self.lidar = self.lidar if len(self.lidar) > 100 else lidar
        self.tree_l = cKDTree(self.lidar)
        self.lo_l, self.hi_l = self.lidar.min(0), self.lidar.max(0)
        self.lidar_score = subset(self.lidar, N_SCORE, 4)
        band = self.lidar[:, [0, 2]]
        self.origin_l = band.min(0) - CELL
        self.shape_l = tuple(int(v) for v in np.ceil((band.max(0) - self.origin_l) / CELL) + 2)
        radius = np.linalg.norm(ours[:, [0, 2]] - ours[:, [0, 2]].mean(0), axis=1).max()
        self.size_o = int(np.ceil(2 * radius * SCALES[-1] * 1.05 / CELL)) + 2
        self.fft_shape = tuple(next_fast_len(n + self.size_o) for n in self.shape_l)
        self.map_l = rfft2(occupancy(band, self.origin_l, self.shape_l), self.fft_shape)
        self.tree_o = cKDTree(ours)
        self.bands: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def band(self, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Our points in the height band at scale s: all, a map subset, a score subset."""
        if s not in self.bands:
            h = s * (self.ours[:, 1] - self.floor_o)
            pts = self.ours[(h > BAND[0]) & (h < self.top)]
            pts = pts if len(pts) > 100 else self.ours
            self.bands[s] = (pts, subset(pts, N_BAND), subset(pts, N_SCORE, 1))
        return self.bands[s]

    def score(self, M: np.ndarray, tol: float = INLIER) -> float:
        """Geometric mean of the inlier fractions both ways; forward only over our points
        that land inside the scan's box (a scan cropped to part of the room must not
        punish the rest of ours), and 0 if fewer than `MIN_INSIDE` of them do."""
        s = float(np.cbrt(np.linalg.det(M[:3, :3])))
        moved = self.band(round(s, 5))[2] @ M[:3, :3].T + M[:3, 3]
        inside = ((moved >= self.lo_l - tol) & (moved <= self.hi_l + tol)).all(1)
        if inside.sum() < MIN_INSIDE:
            return 0.0
        d, _ = self.tree_l.query(moved[inside], distance_upper_bound=tol, workers=-1)
        forward = float(np.isfinite(d).mean())
        back = inlier_fraction(np.linalg.inv(M), self.lidar_score, self.tree_o, tol / s)
        return float(np.sqrt(forward * back))

    def evaluate(self, s: float, yaw: float, tol: float = INLIER) -> dict[str, Any]:
        """Translation for (s, yaw): map correlation (x, z) and floors (y); its score."""
        xz = (s * self.band(s)[1] @ rotation_y(yaw).T)[:, [0, 2]]
        origin_o = xz.min(0)
        map_o = rfft2(occupancy(xz, origin_o, (self.size_o, self.size_o)), self.fft_shape)
        corr = irfft2(self.map_l * np.conj(map_o), self.fft_shape)
        d = np.array(np.unravel_index(int(np.argmax(corr)), corr.shape))
        d = np.where(d < self.shape_l, d, d - np.array(self.fft_shape))  # negative shifts
        t_xz = self.origin_l - origin_o + d * CELL
        M = similarity(s, yaw, np.array([t_xz[0], self.floor_l - s * self.floor_o, t_xz[1]]))
        return {"scale": s, "yaw": yaw, "M": M, "score": self.score(M, tol)}


def pattern_search(search: Search, best: dict[str, Any]) -> dict[str, Any]:
    """Local search around a coarse candidate, scored at 3 cm: move to the best of the 8
    neighbours at steps (2.5 %, 5 deg) while it improves, then halve the steps, down to
    0.3 % / 0.6 deg."""
    ds, dy = 0.025, 5.0
    best = search.evaluate(best["scale"], best["yaw"], FINE)
    while ds > 0.003:
        moves = [search.evaluate(round(best["scale"] * (1 + i * ds), 5), best["yaw"] + j * dy,
                                 FINE) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
        top = max(moves, key=lambda m: m["score"])
        if top["score"] > best["score"]:
            best = top
        else:
            ds, dy = ds / 2, dy / 2
    return best


def distinct(cands: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """The n best candidates that differ by > 15 deg in yaw or > 0.1 in scale."""
    kept: list[dict[str, Any]] = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if all(abs((c["yaw"] - k["yaw"] + 180) % 360 - 180) > 15
               or abs(c["scale"] - k["scale"]) > 0.1 for k in kept):
            kept.append(c)
        if len(kept) == n:
            break
    return kept


def icp(src: np.ndarray, tgt: np.ndarray, tree: cKDTree, M: np.ndarray) -> np.ndarray:
    """Similarity ICP (Umeyama per iteration) on mutual nearest neighbours only: our
    points beyond the edge of a scan that covers less of the room than ours would pull
    the cloud inwards (and scan points beyond ours outwards), biasing the scale."""
    for dist in ICP_STEPS:
        for _ in range(ICP_ITERATIONS):
            moved = src @ M[:3, :3].T + M[:3, 3]
            d, j = tree.query(moved, distance_upper_bound=dist, workers=-1)
            i = np.flatnonzero(np.isfinite(d))
            _, back = cKDTree(moved).query(tgt[j[i]], workers=-1)
            i = i[back == i]
            if len(i) < 100:
                break
            new = sim3_matrix(*umeyama(src[i], tgt[j[i]]))
            change, M = float(np.abs(new - M).max()), new
            if change < 1e-5:
                break
    return M


def decompose(M: np.ndarray) -> tuple[float, float, float]:
    """Scale, yaw about +Y (deg, rotation_y convention) and tilt of +Y (deg)."""
    s = float(np.cbrt(np.linalg.det(M[:3, :3])))
    R = M[:3, :3] / s
    yaw = np.degrees(np.arctan2(R[0, 2] - R[2, 0], R[0, 0] + R[2, 2]))
    return s, float(yaw), float(np.degrees(np.arccos(np.clip(R[1, 1], -1.0, 1.0))))


def fit(ours: np.ndarray, lidar: np.ndarray) -> dict[str, Any]:
    """Sim(3) from our (2 cm) cloud to the (2 cm) scan; statistics of the fit."""
    start = time.perf_counter()
    search = Search(ours, lidar)
    coarse = [search.evaluate(s, yaw) for s in SCALES for yaw in YAWS]
    refined = []
    for c in distinct(coarse, TOP):
        best = pattern_search(search, c)
        src = subset(search.band(best["scale"])[0], N_ICP, 2)
        M = icp(src, search.lidar, search.tree_l, best["M"])
        refined.append((search.score(M, FINE), M, c))
    score, M, c = max(refined, key=lambda r: r[0])
    tree = cKDTree(lidar)
    d, _ = tree.query(ours @ M[:3, :3].T + M[:3, 3], distance_upper_bound=OVERLAP, workers=-1)
    near = d[np.isfinite(d)]
    s, yaw, tilt = decompose(M)
    return {
        "scale": round(s, 5), "yaw_deg": round(yaw, 2), "tilt_deg": round(tilt, 2),
        "translation": np.round(M[:3, 3], 4).tolist(), "matrix": np.round(M, 6).tolist(),
        "score": round(score, 4), "inliers_5cm": round(float(np.mean(d < INLIER)), 4),
        "overlap_10cm": round(float(np.isfinite(d).mean()), 4),
        "residual_median_m": round(float(np.median(near)), 4) if len(near) else None,
        "residual_p90_m": round(float(np.percentile(near, 90)), 4) if len(near) else None,
        "coarse": {"scale": float(c["scale"]), "yaw_deg": float(c["yaw"]),
                   "score": round(c["score"], 4), "candidates": len(coarse)},
        "floors": {"ours": round(search.floor_o, 3), "lidar": round(search.floor_l, 3)},
        "seconds": round(time.perf_counter() - start, 1),
    }


def corrected_changes(out: Path, scale: float) -> list[dict[str, Any]]:
    """Distances (moved) and obb sizes of changes.json, as reported and times `scale`."""
    path = out / "changes.json"
    if not path.is_file():
        return []
    rows = []
    for c in json.loads(path.read_text()).get("changes") or []:
        dist = (c.get("transform") or {}).get("translation_m")
        side = c.get("before") or c.get("after") or {}
        size = (side.get("obb") or {}).get("extents") or side.get("extents")
        rows.append({"id": c.get("id"), "type": c.get("type"), "label": c.get("label"),
                     "distance_m": dist,
                     "distance_metric_m": None if dist is None else round(dist * scale, 3),
                     "size_m": size,
                     "size_metric_m": None if not size else [round(v * scale, 3) for v in size]})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="pipeline output directory")
    parser.add_argument("--before", type=Path, help="LiDAR PLY or OBJ of the before state")
    parser.add_argument("--after", type=Path, help="LiDAR PLY or OBJ of the after state")
    parser.add_argument("--lidar-up", choices=("y", "z"), default="y", help="scan up axis (y)")
    parser.add_argument("--json", action="store_true", help="also write <out>/lidar_check.json")
    args = parser.parse_args(argv)
    scans = {rec: getattr(args, rec) for rec in RECORDINGS if getattr(args, rec)}
    if not scans:
        parser.error("give --before and/or --after")
    meta = args.out_dir / "recon" / "meta.json"
    if meta.is_file() and json.loads(meta.read_text()).get("levelled") is False:
        parser.error("the reconstruction is not levelled (+Y is not up): rerun with "
                     "recon.level_world=true")

    report: dict[str, Any] = {"out_dir": str(args.out_dir), "fits": {}}
    for rec, scan in scans.items():
        ours_raw, source = our_cloud(args.out_dir, rec)
        lidar_raw = read_ply(scan, args.lidar_up)
        ours, lidar = downsample(ours_raw), downsample(lidar_raw)
        print(f"{rec}: ours {source} ({len(ours_raw):,} -> {len(ours):,} points at 2 cm), "
              f"scan {scan} ({len(lidar_raw):,} -> {len(lidar):,})", flush=True)
        del ours_raw, lidar_raw
        f = fit(ours, lidar)
        f.update(ours=source, scan=str(scan))
        report["fits"][rec] = f
        print(f"  scale {f['scale']:.4f} (ours -> metres), yaw {f['yaw_deg']:.1f} deg, tilt "
              f"{f['tilt_deg']:.1f} deg; residual median {f['residual_median_m']} m, p90 "
              f"{f['residual_p90_m']} m over the overlap ({f['overlap_10cm']:.0%} of our points "
              f"within 10 cm, {f['inliers_5cm']:.0%} within 5 cm); {f['seconds']} s")
        if f["overlap_10cm"] < 0.5:
            print("  WARN  poor overlap: does the scan show the same part of the room?")

    scales = [f["scale"] for f in report["fits"].values()]
    scale = float(np.exp(np.mean(np.log(scales))))  # geometric mean
    if len(scales) == 2:
        diff = abs(scales[1] / scales[0] - 1)
        status = "PASS" if diff <= CONSISTENT else "WARN"
        report["consistency"] = {"scale_ratio": round(scales[1] / scales[0], 4),
                                 "differ": round(diff, 4), "status": status}
        print(f"{status}  scale consistency: before {scales[0]:.4f}, after {scales[1]:.4f}, "
              f"differ by {diff:.1%} (want < {CONSISTENT:.0%})")
    report["scale"] = round(scale, 5)
    report["changes"] = corrected_changes(args.out_dir, scale)
    if report["changes"]:
        print(f"\nchanges.json in metres (x {scale:.4f}):\n")
        print("| id | type | label | distance (m) | metric distance (m) | size w x h x d (m) "
              "| metric size (m) |\n|---|---|---|---|---|---|---|")
        for c in report["changes"]:
            dist = "-" if c["distance_m"] is None else f"{c['distance_m']:.2f}"
            metric = "-" if c["distance_metric_m"] is None else f"{c['distance_metric_m']:.2f}"
            size, msize = (" x ".join(f"{v:.2f}" for v in x) if x else "-"
                           for x in (c["size_m"], c["size_metric_m"]))
            print(f"| {c['id']} | {c['type']} | {c['label']} | {dist} | {metric} | {size} | "
                  f"{msize} |")
    if args.json:
        path = args.out_dir / "lidar_check.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
