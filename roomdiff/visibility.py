"""M3 visibility: per-recording occupied / free / unknown voxel grids and change candidates.

For each recording a voxel is *occupied* if enough of its valid depth points fall inside,
*free* if enough of its keyframes looked through it (rays carved from the camera to just
before the observed surface), else *unknown*. Change candidates compare the two grids:

* disappeared: occupied before, free after          (something was removed)
* appeared:    occupied after,  free before         (something was added)
* unverified_before / unverified_after: occupied in one recording, never observed by the
  other (no verdict possible)

Voxels closer than `dist_margin` to a surface of the other recording are ignored
(residual misalignment), the two change masks get a binary opening with a small cube
(`opening_size`, default 2: surfaces are only 2-3 voxels thick, so a 3x3x3 cube erases
real objects too), and 26-connected components above a minimum size become candidate
blobs with evidence counts. Counting and ray carving run in torch (GPU), labelling in
numpy / scipy.

Writes `<out>/visibility/grid.npz`, `candidates.json`, `meta.json` and `debug/*.jpg`
(docs/interfaces.md, M3).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import ndimage

from roomdiff.paths import RECORDINGS, OutputPaths
from roomdiff.projection import best_views, project_aabb, project_to_pixels
from roomdiff.recon import load_recon

if TYPE_CHECKING:
    import torch

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
STATE_NAMES = ("unknown", "free", "occupied")
# candidate type -> (own recording, other recording); own = where the blob is occupied
CANDIDATE_TYPES: dict[str, tuple[int, int]] = {
    "disappeared": (0, 1),
    "appeared": (1, 0),
    "unverified_before": (0, 1),
    "unverified_after": (1, 0),
}
OPENED_TYPES = ("disappeared", "appeared")  # types cleaned by the binary opening
TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "disappeared": (230, 40, 40),
    "appeared": (40, 200, 60),
    "unverified_before": (120, 120, 120),
    "unverified_after": (235, 200, 40),
}
AUTO_VOXEL_RANGE = (0.02, 0.05)  # m, clip range of voxel_size = "auto"
MAX_SAMPLES_PER_CHUNK = 2**23  # ray samples per carving chunk (bounds GPU memory)


# ------------------------------------------------------------------------ grid


@dataclass
class Grid:
    """Axis-aligned voxel grid; voxel (i, j, k) spans `origin + [i, i+1) * voxel_size`
    along x (likewise j / y, k / z). Flat index = (i * ny + j) * nz + k (C order)."""

    origin: np.ndarray
    voxel_size: float
    dims: tuple[int, int, int]

    @property
    def size(self) -> int:
        return int(np.prod(self.dims))

    def voxel_of(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Integer voxel indices (N, 3) of world points and the mask of those inside."""
        ijk = np.floor((np.asarray(points) - self.origin) / self.voxel_size).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < np.asarray(self.dims)), axis=-1)
        return ijk, inside

    def centers(self, ijk: np.ndarray) -> np.ndarray:
        """World coordinates of voxel centres for integer (or fractional) indices (N, 3)."""
        return self.origin + (np.asarray(ijk, np.float64) + 0.5) * self.voxel_size


def resolve_voxel_size(setting: float | str, recon_meta: Mapping[str, Any]) -> tuple[float, str]:
    """Voxel size in metres and how it was chosen.

    "auto": twice the larger co-visible median nearest-neighbour distance between the two
    recordings (from recon `meta.json`), clipped to `AUTO_VOXEL_RANGE`.
    """
    if setting != "auto":
        if isinstance(setting, str):
            raise ValueError(f"visibility.voxel_size must be a number or 'auto', got {setting!r}")
        return float(setting), "fixed"
    covis = recon_meta.get("diagnostics", {}).get("nn_covisible", {})
    medians = [d["median_m"] for d in covis.values() if "median_m" in d]
    if not medians:
        return AUTO_VOXEL_RANGE[0], "auto (no co-visible diagnostics: lower bound)"
    median = max(medians)
    voxel = float(np.clip(2 * median, *AUTO_VOXEL_RANGE))
    return voxel, f"auto (2 x co-visible median NN {100 * median:.2f} cm, clipped)"


def make_grid(
    lo: np.ndarray, hi: np.ndarray, voxel: float, max_voxels: int, pad: int = 3
) -> tuple[Grid, bool]:
    """Grid covering the box [lo, hi] plus `pad` voxels per side. If that needs more than
    `max_voxels` cells the voxel size grows until it fits; returns (grid, grew)."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    grew = False
    while True:
        dims = np.ceil(np.maximum(hi - lo, voxel) / voxel).astype(np.int64) + 2 * pad
        total = float(np.prod(dims.astype(np.float64)))
        if total <= max_voxels:
            break
        voxel *= (total / max_voxels) ** (1 / 3) * 1.001
        grew = True
    origin = lo - pad * voxel
    return Grid(origin, float(voxel), (int(dims[0]), int(dims[1]), int(dims[2]))), grew


# ------------------------------------------------------------------ torch part


def _pixel_dirs(K: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """Camera-frame ray directions (M, 3) with z = 1 through pixels (rows, cols)."""
    import torch

    x = (cols.float() - K[0, 2]) / K[0, 0]
    y = (rows.float() - K[1, 2]) / K[1, 1]
    return torch.stack([x, y, torch.ones_like(x)], dim=-1)


def _flat_index(points: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Flat voxel indices of the world points (M, 3) that fall inside the grid."""
    import torch

    origin = torch.as_tensor(grid.origin, dtype=points.dtype, device=points.device)
    dims = torch.as_tensor(grid.dims, device=points.device)
    ijk = torch.floor((points - origin) / grid.voxel_size).long()
    ijk = ijk[((ijk >= 0) & (ijk < dims)).all(dim=-1)]
    return (ijk[:, 0] * grid.dims[1] + ijk[:, 1]) * grid.dims[2] + ijk[:, 2]


def recording_points_torch(
    recon: Mapping[str, Any], rec_id: int, device: torch.device
) -> torch.Tensor:
    """World points (M, 3) float32 of all valid pixels of one recording, on `device`."""
    import torch

    chunks = [torch.zeros((0, 3), device=device)]
    for i in np.flatnonzero(recon["rec"] == rec_id):
        valid = torch.from_numpy(recon["valid"][i]).to(device)
        rows, cols = torch.nonzero(valid, as_tuple=True)
        depth = torch.from_numpy(recon["depth"][i]).to(device)[rows, cols]
        K = torch.from_numpy(recon["K"][i]).to(device)
        T = torch.from_numpy(recon["cam2world"][i]).to(device)
        cam = _pixel_dirs(K, rows, cols) * depth[:, None]
        chunks.append(cam @ T[:3, :3].T + T[:3, 3])
    return torch.cat(chunks)


def scene_bounds(points: torch.Tensor, percentile: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis [percentile, 100 - percentile] bounds of points (M, 3) (subsampled to at
    most 4M points, since torch.quantile has an input size limit)."""
    import torch

    if len(points) == 0:
        raise ValueError("no valid points in either recording: nothing to voxelise")
    step = max(1, math.ceil(len(points) / 4_000_000))
    q = torch.tensor([percentile / 100, 1 - percentile / 100], device=points.device)
    bounds = torch.quantile(points[::step].double(), q.double(), dim=0).cpu().numpy()
    return bounds[0], bounds[1]


def occupancy_counts(points: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Number of points (M, 3) in each voxel, flat int32 tensor of size `grid.size`."""
    import torch

    counts = torch.zeros(grid.size, dtype=torch.int32, device=points.device)
    index = _flat_index(points, grid)
    counts.index_add_(0, index, torch.ones_like(index, dtype=torch.int32))
    return counts


def free_counts(
    recon: Mapping[str, Any],
    frames: Sequence[int],
    grid: Grid,
    device: torch.device,
    pixel_stride: int = 2,
    carve_rel: float = 0.03,
    carve_abs: float = 0.04,
) -> torch.Tensor:
    """Number of `frames` that saw each voxel empty, flat int16 tensor.

    Rays from every `pixel_stride`-th valid pixel are sampled every voxel_size / 2 from
    the camera centre up to z-depth d * (1 - carve_rel) - carve_abs. Each frame counts at
    most once per voxel.
    """
    import torch

    step = grid.voxel_size / 2
    counts = torch.zeros(grid.size, dtype=torch.int16, device=device)
    seen = torch.zeros(grid.size, dtype=torch.bool, device=device)
    for i in frames:
        valid = torch.from_numpy(recon["valid"][i][::pixel_stride, ::pixel_stride]).to(device)
        depth = torch.from_numpy(recon["depth"][i][::pixel_stride, ::pixel_stride]).to(device)
        rows, cols = torch.nonzero(valid & (depth > 0), as_tuple=True)
        end_z = depth[rows, cols] * (1 - carve_rel) - carve_abs
        keep = end_z > 0
        if not keep.any():
            continue
        K = torch.from_numpy(recon["K"][i]).to(device)
        T = torch.from_numpy(recon["cam2world"][i]).to(device)
        dirs = _pixel_dirs(K, rows[keep] * pixel_stride, cols[keep] * pixel_stride)
        norms = dirs.norm(dim=-1)
        lengths = end_z[keep] * norms  # Euclidean distance to the carving end point
        dirs = (dirs / norms[:, None]) @ T[:3, :3].T
        n_steps = math.ceil(float(lengths.max()) / step) + 1
        t = torch.arange(n_steps, device=device, dtype=torch.float32) * step
        chunk = max(1, MAX_SAMPLES_PER_CHUNK // n_steps)
        seen.zero_()
        for s in range(0, len(lengths), chunk):
            on_ray = t[None, :] <= lengths[s:s + chunk, None]  # (C, S)
            samples = T[:3, 3] + dirs[s:s + chunk, None, :] * t[None, :, None]  # (C, S, 3)
            seen[_flat_index(samples[on_ray], grid)] = True
        counts += seen
    return counts


# ------------------------------------------------------------------ numpy part


def voxel_states(occ_count: np.ndarray, free_count: np.ndarray, n_occ_min: int,
                 n_free_min: int) -> np.ndarray:
    """uint8 states: OCCUPIED if occ_count >= n_occ_min, else FREE if free_count >=
    n_free_min, else UNKNOWN."""
    state = np.full(occ_count.shape, UNKNOWN, np.uint8)
    state[free_count >= n_free_min] = FREE
    state[occ_count >= n_occ_min] = OCCUPIED
    return state


def far_from_surfaces(occupied: np.ndarray, voxel: float, margin: float) -> np.ndarray:
    """Voxels whose Euclidean distance (m) to the nearest occupied voxel exceeds `margin`."""
    if not occupied.any():
        return np.ones(occupied.shape, bool)
    return ndimage.distance_transform_edt(~occupied, sampling=voxel) > margin


def candidate_masks(
    states: Sequence[np.ndarray], far: Sequence[np.ndarray]
) -> dict[str, np.ndarray]:
    """Raw candidate voxels per type from the two state grids and the "far from the
    other recording's surfaces" masks (both indexed 0 = before, 1 = after)."""
    masks = {}
    for kind, (own, other) in CANDIDATE_TYPES.items():
        wanted = UNKNOWN if kind.startswith("unverified") else FREE
        masks[kind] = (states[own] == OCCUPIED) & (states[other] == wanted) & far[other]
    return masks


def label_blobs(
    mask: np.ndarray, voxel: float, opening: int, min_volume: float, min_voxels: int
) -> tuple[np.ndarray, list[int], int]:
    """Optional binary opening (cube of edge `opening` voxels; < 2 = off), then
    26-connected labelling. Returns (labels int32, kept component labels, voxels left
    after the opening)."""
    if opening >= 2:
        mask = ndimage.binary_opening(mask, structure=np.ones((opening,) * 3, bool))
    labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = (sizes >= min_voxels) & (sizes * voxel**3 >= min_volume)
    keep[0] = False
    return labels.astype(np.int32, copy=False), np.flatnonzero(keep).tolist(), int(mask.sum())


def blob_record(
    labels: np.ndarray, comp: int, box: tuple[slice, ...], kind: str, grid: Grid,
    occ: Sequence[np.ndarray], free: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Geometry and voxel evidence of one labelled component (`box` from find_objects)."""
    own, other = CANDIDATE_TYPES[kind]
    inside = labels[box] == comp
    ijk = np.argwhere(inside) + [s.start for s in box]
    start = np.array([s.start for s in box])
    stop = np.array([s.stop for s in box])
    count = len(ijk)
    return {
        "type": kind,
        "voxels": count,
        "volume_m3": round(count * grid.voxel_size**3, 6),
        "centroid": grid.centers(ijk.mean(0)).round(4).tolist(),
        "aabb": [(grid.origin + start * grid.voxel_size).round(4).tolist(),
                 (grid.origin + stop * grid.voxel_size).round(4).tolist()],
        "evidence": {
            "free_frames_mean": round(float(free[other][box][inside].mean()), 2),
            "occ_points": int(occ[own][box][inside].sum()),
        },
    }


def view_evidence(
    centroids: np.ndarray, recon: Mapping[str, Any], rec_id: int, voxel: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per centroid: number of keyframes of `rec_id` that saw past it (inside the image,
    valid pixel, depth >= centroid depth + voxel) and the distance to its nearest
    camera centre."""
    frames = np.flatnonzero(recon["rec"] == rec_id)
    height, width = recon["depth"].shape[1:]
    views = np.zeros(len(centroids), np.int64)
    for i in frames:
        col, row, z, inside = project_to_pixels(centroids, recon["K"][i], recon["cam2world"][i],
                                                width, height)
        idx = np.flatnonzero(inside)
        depth = recon["depth"][i][row[idx], col[idx]]
        seen = recon["valid"][i][row[idx], col[idx]] & (depth >= z[idx] + voxel)
        views[idx[seen]] += 1
    centres = recon["cam2world"][frames, :3, 3].astype(np.float64)
    if len(centres) == 0 or len(centroids) == 0:
        return views, np.full(len(centroids), np.inf)
    dist = np.linalg.norm(centroids[:, None, :] - centres[None], axis=-1).min(1)
    return views, dist


# ---------------------------------------------------------------- debug crops


def write_debug_crops(
    debug_dir: Path, blobs: list[dict[str, Any]], recon: Mapping[str, Any],
    n_views: int = 2, occlusion_tol: float = 0.10,
) -> int:
    """For each blob and recording, draw the blob's projected AABB on the `n_views`
    keyframes that see its centroid best; adds `views` to each blob. Returns the number
    of images written."""
    from PIL import Image, ImageDraw, ImageFont

    debug_dir.mkdir(parents=True, exist_ok=True)
    for old in debug_dir.glob("blob_*.jpg"):
        old.unlink()
    height, width = recon["depth"].shape[1:]
    font = ImageFont.load_default(size=14)
    written = 0
    for blob in blobs:
        color = TYPE_COLORS[blob["type"]]
        blob["views"] = {}
        for rec_id, name in enumerate(RECORDINGS):
            frames = np.flatnonzero(recon["rec"] == rec_id)
            chosen = best_views(blob["centroid"], frames, recon["cam2world"], recon["K"],
                                recon["depth"], n_views, occlusion_tol)
            blob["views"][name] = []
            for k, (i, occluded) in enumerate(chosen):
                image = Image.fromarray(recon["images"][i])
                draw = ImageDraw.Draw(image)
                box = project_aabb(*blob["aabb"], recon["K"][i], recon["cam2world"][i],
                                   width, height)
                if box is not None:
                    draw.rectangle(box, outline=color, width=3)
                text = f"{blob['id']}:{blob['type']} {name} {recon['files'][i]}"
                if occluded:
                    text += " (occluded)"
                draw.rectangle(draw.textbbox((4, 4), text, font=font), fill=(0, 0, 0))
                draw.text((4, 4), text, fill=color, font=font)
                file = f"blob_{blob['id']}_{blob['type']}_{name}_{k}.jpg"
                image.save(debug_dir / file, quality=85)
                written += 1
                blob["views"][name].append(
                    {"frame": int(i), "file": str(recon["files"][i]), "occluded": occluded,
                     "box": list(box) if box is not None else None, "crop": f"debug/{file}"})
    return written


# ---------------------------------------------------------------------- stage


def _device(name: str) -> torch.device:
    import torch

    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[visibility] CUDA not available, running on the CPU")
        name = "cpu"
    return torch.device(name)


def load_grid(path: str | Path) -> dict[str, Any]:
    """Read `grid.npz` into a dict; `voxel_size` becomes a float."""
    with np.load(path) as f:
        grid = {key: f[key] for key in f.files}
    grid["voxel_size"] = float(grid["voxel_size"])
    return grid


def run_visibility(out_dir: str | Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Build both voxel grids from `<out>/recon/` and write `<out>/visibility/`.

    `cfg` is the full config; only its `visibility` section is used. Returns `meta.json`.
    """
    import torch

    params = dict(cfg["visibility"])
    paths = OutputPaths(out_dir)
    out = paths.visibility
    times: dict[str, float] = {}
    start = time.perf_counter()

    recon = load_recon(paths.recon / "recon.npz")
    meta_path = paths.recon / "meta.json"
    recon_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    voxel, voxel_source = resolve_voxel_size(params["voxel_size"], recon_meta)
    times["load"] = time.perf_counter() - start

    t0 = time.perf_counter()
    device = _device(params["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    frames = [np.flatnonzero(recon["rec"] == r) for r in range(len(RECORDINGS))]
    with torch.no_grad():
        points = [recording_points_torch(recon, r, device) for r in range(len(RECORDINGS))]
        lo, hi = scene_bounds(torch.cat(points), params["bounds_percentile"])
        grid, grew = make_grid(lo, hi, voxel, params["max_voxels"], params["pad_voxels"])
        if grew:
            print(f"[visibility] grid would exceed max_voxels={params['max_voxels']}: voxel "
                  f"size increased from {voxel:.4f} to {grid.voxel_size:.4f} m")
        occ_t = [occupancy_counts(p, grid) for p in points]
        n_points = [len(p) for p in points]
        del points
        free_t = [free_counts(recon, f, grid, device, params["pixel_stride"],
                              params["carve_rel"], params["carve_abs"]) for f in frames]
        occ = [c.cpu().numpy().reshape(grid.dims) for c in occ_t]
        free = [c.cpu().numpy().reshape(grid.dims) for c in free_t]
    del occ_t, free_t
    peak = None
    if device.type == "cuda":
        peak = {"allocated": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                "reserved": round(torch.cuda.max_memory_reserved(device) / 2**30, 2)}
        torch.cuda.empty_cache()
    times["voxelise_and_carve"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    states = [voxel_states(o, f, params["n_occ_min"], params["n_free_min"])
              for o, f in zip(occ, free)]
    far = [far_from_surfaces(s == OCCUPIED, grid.voxel_size, params["dist_margin"])
           for s in states]
    raw = candidate_masks(states, far)
    del far
    times["distance_and_candidates"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    blobs, labelled, candidate_voxels = [], {}, {}
    for kind, mask in raw.items():
        opening = params["opening_size"] if kind in OPENED_TYPES else 0
        labels, kept, after_opening = label_blobs(
            mask, grid.voxel_size, opening, params["min_blob_volume"], params["min_blob_voxels"])
        boxes = ndimage.find_objects(labels)
        for comp in kept:
            blob = blob_record(labels, comp, boxes[comp - 1], kind, grid, occ, free)
            blobs.append((comp, blob))
        labelled[kind] = labels
        candidate_voxels[kind] = {"raw": int(mask.sum()), "after_opening": after_opening}
    del raw
    blobs.sort(key=lambda cb: -cb[1]["volume_m3"])
    luts = {kind: np.zeros(int(labels.max()) + 1, np.int32) for kind, labels in labelled.items()}
    records = []
    for blob_id, (comp, blob) in enumerate(blobs, start=1):
        luts[blob["type"]][comp] = blob_id
        records.append({"id": blob_id, **blob})
    for kind, labels in labelled.items():
        labelled[kind] = luts[kind][labels]
        candidate_voxels[kind]["in_blobs"] = int((labelled[kind] > 0).sum())
    times["labelling"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for kind, (_, other) in CANDIDATE_TYPES.items():
        sel = [r for r in records if r["type"] == kind]
        if not sel:
            continue
        centroids = np.array([r["centroid"] for r in sel], np.float64)
        views, dist = view_evidence(centroids, recon, other, grid.voxel_size)
        for r, n, d in zip(sel, views, dist):
            r["evidence"]["observing_views"] = int(n)
            r["evidence"]["min_camera_distance"] = round(float(d), 3)
    debug_blobs = records[:params["max_debug_blobs"]]
    n_crops = write_debug_crops(out / "debug", debug_blobs, recon, params["debug_views"],
                                params["occlusion_tol"])
    times["evidence_and_crops"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for r, name in enumerate(RECORDINGS):
        arrays[f"state_{name}"] = states[r]
        arrays[f"free_count_{name}"] = free[r]
        arrays[f"occ_count_{name}"] = occ[r]
    for kind, labels in labelled.items():
        arrays[f"labels_{kind}"] = labels
    np.savez_compressed(out / "grid.npz", voxel_size=np.float64(grid.voxel_size),
                        origin=grid.origin, dims=np.array(grid.dims), **arrays)
    blob_counts = {kind: sum(r["type"] == kind for r in records) for kind in CANDIDATE_TYPES}
    candidates = {"voxel_size": grid.voxel_size, "origin": grid.origin.round(5).tolist(),
                  "dims": list(grid.dims), "counts": blob_counts, "candidates": records}
    (out / "candidates.json").write_text(json.dumps(candidates, indent=1))
    times["write"] = time.perf_counter() - t0
    times["total"] = time.perf_counter() - start

    state_counts = {name: {s: int((states[r] == v).sum()) for v, s in enumerate(STATE_NAMES)}
                    for r, name in enumerate(RECORDINGS)}
    meta = {
        "params": params,
        "voxel_size": grid.voxel_size,
        "voxel_size_source": voxel_source + (" -> enlarged to fit max_voxels" if grew else ""),
        "origin": grid.origin.round(5).tolist(),
        "dims": list(grid.dims),
        "n_voxels": grid.size,
        "bounds": [lo.round(4).tolist(), hi.round(4).tolist()],
        "device": str(device),
        "points": dict(zip(RECORDINGS, n_points)),
        "voxels_per_state": state_counts,
        "candidate_voxels": candidate_voxels,
        "blobs_per_type": blob_counts,
        "debug_crops": n_crops,
        "runtime_s": {k: round(v, 2) for k, v in times.items()},
        "peak_vram_gb": peak,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    vram = f", peak VRAM {peak['allocated']:.2f} GB" if peak else ""
    print(f"[visibility] grid {'x'.join(map(str, grid.dims))} at {100 * grid.voxel_size:.2f} cm"
          f" ({voxel_source}), {times['total']:.1f} s{vram} -> {out}")
    print("[visibility] blobs: " + ", ".join(f"{k} {v}" for k, v in blob_counts.items()))
    return meta
