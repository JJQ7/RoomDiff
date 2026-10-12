"""M4 instances: object instances from the M3 change blobs, per recording.

Each selected blob (all disappeared / appeared blobs and the large unverified ones) is
handled in its *owner* recording (where it is occupied):

1. keyframes that see the blob well are chosen (centroid in view, blob not occluded,
   large projected box, varied viewpoints);
2. SAM 2.1 is prompted there with the blob's projected box and its depth-visible voxel
   centres; the best mask that stays inside the box is kept;
3. the mask pixels are lifted to 3D with the reconstructed depth and filtered to the
   blob's neighbourhood (static surfaces shared with the other recording are dropped);
4. fragments of one object are merged (shared-view mask IoU or 3D overlap);
5. each instance gets a DINOv3 feature (mask-pooled patch tokens, averaged over views)
   and a CLS token of its box crop, and SAM masks in every keyframe of its recording
   where its points are visible (COCO RLE, for the SceneDiff evaluation).

Writes `<out>/instances/<rec>.json`, `<rec>_feats.npy`, `<rec>_cls.npy`,
`<rec>_points.npz`, `<rec>_masks.npz` and `meta.json` (docs/interfaces.md, M4).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from roomdiff.geometry import floor_obb, unproject_depth, voxel_downsample
from roomdiff.models import Featurizer, Segmenter
from roomdiff.paths import RECORDINGS, OutputPaths
from roomdiff.projection import best_views, project_aabb, project_to_pixels
from roomdiff.recon import load_recon
from roomdiff.visibility import CANDIDATE_TYPES, OCCUPIED, load_grid

VERIFIED_TYPES = ("disappeared", "appeared")
MIN_MASK_IN_BOX = 0.5  # a SAM mask must have at least this fraction of its pixels in the box
MIN_VISIBLE_FRAC = 0.15  # a view must see at least this fraction of the blob unoccluded
MAX_PROJECTED = 2000  # points projected per view test (random subset)
OUTLIER_RADIUS = 0.03  # m; lifted points with fewer than OUTLIER_MIN_NEIGHBOURS other ...
OUTLIER_MIN_NEIGHBOURS = 3  # ... points within this radius are dropped

PointTest = Callable[[np.ndarray], np.ndarray]  # world points (M, 3) -> bool (M,)


@dataclass
class View:
    """One SAM mask of an instance in a keyframe of its recording (recon resolution)."""

    frame: int
    box: tuple[int, int, int, int]  # prompt box x0, y0, x1, y1
    mask: np.ndarray  # (H, W) bool
    score: float


@dataclass
class Instance:
    """A 3D object hypothesis in one recording (0 = before, 1 = after)."""

    rec: int
    kind: str  # "disappeared" | "appeared" | "unverified"
    blobs: list[dict[str, Any]]  # source M3 blob records
    voxels: np.ndarray  # (V, 3) centres of the source blob voxels
    points: np.ndarray  # (M, 3) float32 lifted object points (1 cm voxels)
    colors: np.ndarray  # (M, 3) uint8
    views: list[View]
    id: int = 0
    feature: np.ndarray | None = None
    cls: np.ndarray | None = None
    view_consistency: float | None = None
    frame_masks: list[tuple[int, dict[str, Any], float]] = field(default_factory=list)

    @property
    def blob_ids(self) -> list[int]:
        return sorted(b["id"] for b in self.blobs)


# ------------------------------------------------------------------ blob choice


def box_distance(point: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> float:
    """Euclidean distance from a point to an axis-aligned box (0 inside)."""
    p, lo, hi = (np.asarray(v, np.float64) for v in (point, lo, hi))
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - p, p - hi))))


def select_blobs(
    candidates: Sequence[Mapping[str, Any]], min_unverified_volume: float,
    hidden_surface_margin: float, max_unverified_extent: float = np.inf,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Blobs that become instance candidates, and the dropped ones with a reason.

    Kept: every disappeared / appeared blob; unverified blobs of at least
    `min_unverified_volume` m^3, with no AABB side above `max_unverified_extent` (larger
    ones are room regions that only one video filmed, not objects), whose centroid is
    farther than `hidden_surface_margin` from every disappeared / appeared AABB (closer
    ones are the surface that a removed or added object hid or uncovered).
    """
    verified = [b for b in candidates if b["type"] in VERIFIED_TYPES]
    kept, dropped = [], []
    for blob in candidates:
        lo, hi = np.asarray(blob["aabb"], np.float64)
        if blob["type"] in VERIFIED_TYPES:
            kept.append(dict(blob))
        elif blob["volume_m3"] < min_unverified_volume:
            dropped.append({"blob": blob["id"], "reason": "small unverified"})
        elif np.max(hi - lo) > max_unverified_extent:
            dropped.append({"blob": blob["id"], "reason": "unverified region too large"})
        elif any(box_distance(blob["centroid"], *v["aabb"]) <= hidden_surface_margin
                 for v in verified):
            dropped.append({"blob": blob["id"], "reason": "hidden surface"})
        else:
            kept.append(dict(blob))
    return kept, dropped


def blob_voxel_centres(grid: Mapping[str, Any], blob: Mapping[str, Any]) -> np.ndarray:
    """World centres (V, 3) of the voxels labelled with this blob in `grid.npz`."""
    ijk = np.argwhere(grid[f"labels_{blob['type']}"] == blob["id"])
    return grid["origin"] + (ijk + 0.5) * grid["voxel_size"]


def static_surface_test(grid: Mapping[str, Any], other: int) -> PointTest:
    """Point test: inside a voxel occupied by recording `other` (dilated by one voxel),
    i.e. on a surface that both recordings share (floor, wall, unchanged furniture)."""
    occupied = ndimage.binary_dilation(grid[f"state_{RECORDINGS[other]}"] == OCCUPIED,
                                       structure=np.ones((3, 3, 3), bool))
    origin, voxel, dims = grid["origin"], grid["voxel_size"], np.asarray(occupied.shape)

    def test(points: np.ndarray) -> np.ndarray:
        ijk = np.floor((np.asarray(points) - origin) / voxel).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < dims), axis=1)
        out = np.zeros(len(ijk), bool)
        out[inside] = occupied[tuple(ijk[inside].T)]
        return out

    return test


# ------------------------------------------------------------------ views and SAM


def _subsample(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(points) <= n:
        return points
    return points[np.random.default_rng(seed).choice(len(points), n, replace=False)]


def project_with_depth(
    points: np.ndarray, frame: int, recon: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pixels (col, row), point depth, in-image mask, observed depth and validity of `points` in
    keyframe `frame` (observed depth / validity are 0 / False outside the image)."""
    height, width = recon["depth"].shape[1:]
    col, row, z, inside = project_to_pixels(points, recon["K"][frame],
                                            recon["cam2world"][frame], width, height)
    observed = np.where(inside, recon["depth"][frame][row, col], 0.0)
    valid = inside & recon["valid"][frame][row, col]
    return col, row, z, inside, np.where(valid, observed, 0.0), valid


def select_views(
    points: np.ndarray, frames: Sequence[int], recon: Mapping[str, Any], n_views: int,
    occlusion_tol: float, min_angle_deg: float,
) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Up to `n_views` keyframes `(frame, projected AABB box)` that see `points` well.

    Candidates come from `best_views` (centroid inside the image and not hidden behind a
    surface farther than half the box diagonal in front of it). They are ranked by
    projected box area x fraction of points not occluded (observed depth >= point depth -
    `occlusion_tol`), then picked greedily so that the viewing directions differ by at
    least `min_angle_deg`; remaining slots are filled by rank.
    """
    if len(points) == 0 or len(frames) == 0:
        return []
    height, width = recon["depth"].shape[1:]
    lo, hi = points.min(0), points.max(0)
    centroid = points.mean(0)
    tol = occlusion_tol + 0.5 * float(np.linalg.norm(hi - lo))
    seen = best_views(centroid, frames, recon["cam2world"], recon["K"], recon["depth"],
                      len(frames), tol)
    sample = _subsample(points, MAX_PROJECTED)
    ranked = []
    for frame, occluded in seen:
        box = project_aabb(lo, hi, recon["K"][frame], recon["cam2world"][frame], width, height)
        if occluded or box is None:
            continue
        _, _, z, inside, depth, _ = project_with_depth(sample, frame, recon)
        visible = float((inside & (depth >= z - occlusion_tol)).mean())
        if visible < MIN_VISIBLE_FRAC:
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        ranked.append((area * visible, frame, box))
    ranked.sort(key=lambda r: -r[0])
    directions = {}
    for _, frame, _ in ranked:
        d = recon["cam2world"][frame][:3, 3] - centroid
        directions[frame] = d / max(np.linalg.norm(d), 1e-9)
    min_cos = np.cos(np.radians(min_angle_deg))
    chosen: list[tuple[int, tuple[int, int, int, int]]] = []
    for _, frame, box in ranked:
        if len(chosen) < n_views and all(directions[frame] @ directions[f] < min_cos
                                         for f, _ in chosen):
            chosen.append((frame, box))
    for _, frame, box in ranked:
        if len(chosen) < n_views and all(frame != f for f, _ in chosen):
            chosen.append((frame, box))
    return chosen


def enlarge_box(
    box: Sequence[float], factor: float, width: int, height: int, min_size: float = 0.0
) -> tuple[int, int, int, int]:
    """Box scaled by `1 + factor` about its centre (each side at least `min_size`),
    shifted / clipped to stay inside the image."""
    x0, y0, x1, y1 = map(float, box)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = min(max((x1 - x0) * (1 + factor), min_size), width - 1)
    h = min(max((y1 - y0) * (1 + factor), min_size), height - 1)
    x0 = float(np.clip(cx - w / 2, 0, width - 1 - w))
    y0 = float(np.clip(cy - h / 2, 0, height - 1 - h))
    return round(x0), round(y0), round(x0 + w), round(y0 + h)


def prompt_points(cols: np.ndarray, rows: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Up to `k` pixels (x, y) spread over the given ones: the pixel nearest to each
    k-means centre (interior points rather than extremes)."""
    pts = np.stack([cols, rows], 1).astype(np.float64)
    pts = np.unique(pts, axis=0)
    if len(pts) <= k:
        return pts
    from scipy.cluster.vq import kmeans2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centres, _ = kmeans2(pts, k, minit="++", seed=seed)
    nearest = cKDTree(pts).query(centres)[1]
    return pts[np.unique(nearest)]


def pick_mask(
    masks: np.ndarray, scores: np.ndarray, box: Sequence[int]
) -> tuple[np.ndarray, float] | None:
    """Highest-scoring non-empty mask with at least `MIN_MASK_IN_BOX` of its pixels inside
    `box`, or None."""
    x0, y0, x1, y1 = box
    best = None
    for mask, score in zip(masks, scores):
        area = int(mask.sum())
        if area == 0 or mask[y0:y1 + 1, x0:x1 + 1].sum() < MIN_MASK_IN_BOX * area:
            continue
        if best is None or score > best[1]:
            best = (mask, float(score))
    return best


def segment_view(
    segmenter: Segmenter, recon: Mapping[str, Any], frame: int, voxels: np.ndarray,
    box: Sequence[int], box_enlarge: float, n_points: int, depth_tol: float,
) -> View | None:
    """SAM mask of the blob in one keyframe: prompt = projected box enlarged by
    `box_enlarge` plus up to `n_points` projected voxel centres lying on the observed
    surface (|observed depth - depth| <= `depth_tol`)."""
    height, width = recon["depth"].shape[1:]
    prompt_box = enlarge_box(box, box_enlarge, width, height)
    col, row, z, _, depth, valid = project_with_depth(voxels, frame, recon)
    on_surface = valid & (np.abs(depth - z) <= depth_tol)
    points = prompt_points(col[on_surface], row[on_surface], n_points) if on_surface.any() \
        else None
    masks, scores = segmenter.segment(str(frame), recon["images"][frame], prompt_box, points)
    picked = pick_mask(masks, scores, prompt_box)
    if picked is None:
        return None
    return View(int(frame), prompt_box, picked[0], picked[1])


# ------------------------------------------------------------------ lifting


def lift_mask(
    recon: Mapping[str, Any], frame: int, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World points and colours of the valid-depth pixels inside `mask`."""
    sel = mask & recon["valid"][frame]
    world = unproject_depth(recon["depth"][frame], recon["K"][frame], recon["cam2world"][frame])
    return world[sel], recon["images"][frame][sel]


def keep_near_blob(
    points: np.ndarray, voxels: np.ndarray, expand_m: float, near_voxel_m: float,
    on_static: PointTest | None = None,
) -> np.ndarray:
    """Which lifted points belong to the object: within `near_voxel_m` of a blob voxel
    centre, or inside the blob AABB expanded by `expand_m` and not on a static surface
    (`on_static`: a surface the other recording shares, e.g. floor or wall). Stops masks
    that bleed onto the background while letting SAM complete fragmentary blobs."""
    if len(points) == 0 or len(voxels) == 0:
        return np.zeros(len(points), bool)
    lo, hi = voxels.min(0) - expand_m, voxels.max(0) + expand_m
    in_box = np.all((points >= lo) & (points <= hi), axis=1)
    keep = in_box.copy()
    if on_static is not None and in_box.any():
        keep[in_box] = ~on_static(points[in_box])
    idx = np.flatnonzero(~keep)
    if len(idx):
        dist, _ = cKDTree(voxels).query(points[idx], distance_upper_bound=near_voxel_m)
        keep[idx[np.isfinite(dist)]] = True
    return keep


def clean_points(
    points: np.ndarray, colors: np.ndarray, voxel: float
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-downsample, then drop isolated points (radius outlier removal)."""
    points, colors = voxel_downsample(points, colors, voxel)
    assert colors is not None
    if len(points) == 0:
        return points, colors
    counts = cKDTree(points).query_ball_point(points, OUTLIER_RADIUS, return_length=True)
    keep = counts > OUTLIER_MIN_NEIGHBOURS  # the point itself is counted
    return points[keep], colors[keep]


def build_instance(
    blob: Mapping[str, Any], voxels: np.ndarray, recon: Mapping[str, Any],
    frames: Sequence[int], segmenter: Segmenter, params: Mapping[str, Any],
    on_static: PointTest | None = None,
) -> tuple[Instance | None, str]:
    """Instance of one blob (views, masks, lifted points) or None with the reason."""
    own = CANDIDATE_TYPES[blob["type"]][0]
    kind = blob["type"] if blob["type"] in VERIFIED_TYPES else "unverified"
    chosen = select_views(voxels, frames, recon, params["n_views"], params["occlusion_tol"],
                          params["min_view_angle"])
    if not chosen:
        return None, "no clear view"
    views, points, colors = [], [], []
    for frame, box in chosen:
        view = segment_view(segmenter, recon, frame, voxels, box, params["box_enlarge"],
                            params["n_prompt_points"], params["occlusion_tol"])
        if view is None:
            continue
        p, c = lift_mask(recon, frame, view.mask)
        keep = keep_near_blob(p, voxels, params["expand_m"], params["near_voxel_m"], on_static)
        views.append(view)
        points.append(p[keep])
        colors.append(c[keep])
    if not views:
        return None, "no SAM mask inside the box"
    pts, cols = clean_points(np.concatenate(points), np.concatenate(colors),
                             params["lift_voxel"])
    if len(pts) < params["min_points"]:
        return None, f"{len(pts)} points < min_points"
    return Instance(own, kind, [dict(blob)], voxels, pts, cols, views), "ok"


# ------------------------------------------------------------------ merging


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def point_overlap(a: np.ndarray, b: np.ndarray, dist: float) -> float:
    """Larger of the two fractions of points lying within `dist` of the other set."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    fa = np.isfinite(cKDTree(b).query(a, distance_upper_bound=dist)[0]).mean()
    fb = np.isfinite(cKDTree(a).query(b, distance_upper_bound=dist)[0]).mean()
    return float(max(fa, fb))


def should_merge(a: Instance, b: Instance, iou: float, dist: float, frac: float) -> bool:
    """Same object: masks overlap (IoU > `iou`) in a keyframe both use, or more than
    `frac` of the points of one lie within `dist` of the other."""
    views_b = {v.frame: v for v in b.views}
    for v in a.views:
        if v.frame in views_b and mask_iou(v.mask, views_b[v.frame].mask) > iou:
            return True
    return point_overlap(a.points, b.points, dist) > frac


def combine(group: Sequence[Instance], voxel: float) -> Instance:
    """One instance from several fragments of the same recording."""
    if len(group) == 1:
        return group[0]
    views: dict[int, View] = {}
    for inst in group:
        for v in inst.views:
            if v.frame in views:
                old = views[v.frame]
                box = (min(old.box[0], v.box[0]), min(old.box[1], v.box[1]),
                       max(old.box[2], v.box[2]), max(old.box[3], v.box[3]))
                views[v.frame] = View(v.frame, box, old.mask | v.mask, max(old.score, v.score))
            else:
                views[v.frame] = v
    points, colors = voxel_downsample(np.concatenate([i.points for i in group]),
                                      np.concatenate([i.colors for i in group]), voxel)
    assert colors is not None
    kinds = [i.kind for i in group]
    kind = next((k for k in VERIFIED_TYPES if k in kinds), "unverified")
    return Instance(group[0].rec, kind, [b for i in group for b in i.blobs],
                    np.concatenate([i.voxels for i in group]), points, colors,
                    sorted(views.values(), key=lambda v: -v.score))


def merge_instances(
    instances: Sequence[Instance], iou: float, dist: float, frac: float, voxel: float
) -> list[Instance]:
    """Union-find merge of the fragments of one recording (see `should_merge`)."""
    parent = list(range(len(instances)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            if root(i) != root(j) and should_merge(instances[i], instances[j], iou, dist, frac):
                parent[root(j)] = root(i)
    groups: dict[int, list[Instance]] = {}
    for i, inst in enumerate(instances):
        groups.setdefault(root(i), []).append(inst)
    return [combine(g, voxel) for g in groups.values()]


# ------------------------------------------------------------------ features


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def pooled_feature(tokens: np.ndarray, mask: np.ndarray, erode_px: int) -> np.ndarray:
    """L2-normalised mean of the patch tokens (gh, gw, D) whose patch centre lies inside
    the eroded mask (falls back to the plain mask, then to the patch at the mask centroid)."""
    gh, gw = tokens.shape[:2]
    height, width = mask.shape
    rows = ((np.arange(gh) + 0.5) * height / gh).astype(int)
    cols = ((np.arange(gw) + 0.5) * width / gw).astype(int)
    eroded = ndimage.binary_erosion(mask, iterations=erode_px) if erode_px > 0 else mask
    for m in (eroded, mask):
        sel = m[np.ix_(rows, cols)]
        if sel.any():
            return _normalise(tokens[sel].mean(0))
    r, c = np.argwhere(mask).mean(0) if mask.any() else (height / 2, width / 2)
    return _normalise(tokens[min(int(r * gh / height), gh - 1), min(int(c * gw / width), gw - 1)])


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight box (x0, y0, x1, y1) of a non-empty mask."""
    rows, cols = np.nonzero(mask)
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def mean_pairwise_cosine(features: Sequence[np.ndarray]) -> float | None:
    """Mean cosine over all pairs of unit vectors; None for fewer than two."""
    if len(features) < 2:
        return None
    f = np.stack(features)
    sims = f @ f.T
    n = len(f)
    return float((sims.sum() - np.trace(sims)) / (n * (n - 1)))


def describe_features(
    inst: Instance, recon: Mapping[str, Any], featurizer: Featurizer, erode_px: int,
    cls_enlarge: float,
) -> None:
    """Set `feature` (mean of per-view mask-pooled tokens), `view_consistency` and `cls`
    (CLS token of the best view's mask box enlarged by `cls_enlarge`)."""
    height, width = recon["depth"].shape[1:]
    per_view = [pooled_feature(featurizer.patch_tokens(str(v.frame), recon["images"][v.frame]),
                               v.mask, erode_px) for v in inst.views]
    inst.feature = _normalise(np.mean(per_view, 0)).astype(np.float32)
    inst.view_consistency = mean_pairwise_cosine(per_view)
    best = max(inst.views, key=lambda v: v.score)
    x0, y0, x1, y1 = enlarge_box(mask_box(best.mask), cls_enlarge, width, height)
    crop = recon["images"][best.frame][y0:y1 + 1, x0:x1 + 1]
    inst.cls = _normalise(featurizer.cls_token(crop)).astype(np.float32)


# ------------------------------------------------------------------ all-frame masks


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    """COCO RLE `{"size": [H, W], "counts": str}` of a bool mask."""
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(s) for s in rle["size"]], "counts": rle["counts"].decode("ascii")}


def decode_rle(rle: Mapping[str, Any]) -> np.ndarray:
    """Bool mask (H, W) of a COCO RLE with str or bytes counts."""
    from pycocotools import mask as mask_utils

    counts = rle["counts"]
    data = {"size": list(rle["size"]),
            "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
    with warnings.catch_warnings():  # pycocotools' numpy-2 `copy=False` deprecation
        warnings.simplefilter("ignore", DeprecationWarning)
        return mask_utils.decode(data).astype(bool)


def frame_masks(
    inst: Instance, frames: Sequence[int], recon: Mapping[str, Any], segmenter: Segmenter,
    visible_frac: float, depth_tol: float, n_points: int, box_enlarge: float,
) -> list[tuple[int, dict[str, Any], float]]:
    """SAM masks `(frame, RLE, score)` of the instance in every keyframe where at least
    `visible_frac` of its points are depth-consistent (|observed - projected depth| <=
    `depth_tol`); prompt = up to `n_points` of those pixels plus their box."""
    height, width = recon["depth"].shape[1:]
    sample = _subsample(inst.points, MAX_PROJECTED)
    out = []
    for frame in frames:
        col, row, z, _, depth, valid = project_with_depth(sample, frame, recon)
        consistent = valid & (np.abs(depth - z) <= depth_tol)
        if consistent.mean() < visible_frac:
            continue
        c, r = col[consistent], row[consistent]
        box = enlarge_box((c.min(), r.min(), c.max(), r.max()), box_enlarge, width, height)
        masks, scores = segmenter.segment(str(frame), recon["images"][frame], box,
                                          prompt_points(c, r, n_points))
        picked = pick_mask(masks, scores, box)
        if picked is not None:
            out.append((int(frame), encode_rle(picked[0]), picked[1]))
    return out


# ------------------------------------------------------------------ records and I/O


def aggregate_evidence(blobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evidence of an instance from its source blobs (verified ones if there are any):
    voxel-weighted `free_frames_mean`, summed `free_rays` (= keyframe x voxel pairs the
    other recording saw empty), max `observing_views`, summed `occ_points`."""
    use = [b for b in blobs if b["type"] in VERIFIED_TYPES] or list(blobs)
    voxels = sum(b["voxels"] for b in use)
    free_rays = sum(b["evidence"]["free_frames_mean"] * b["voxels"] for b in use)
    distances = [b["evidence"].get("min_camera_distance") for b in use]
    distances = [d for d in distances if d is not None]
    return {
        "voxels": int(voxels),
        "volume_m3": round(sum(b["volume_m3"] for b in use), 6),
        "free_rays": round(free_rays),
        "free_frames_mean": round(free_rays / max(voxels, 1), 2),
        "observing_views": int(max(b["evidence"].get("observing_views", 0) for b in use)),
        "occ_points": int(sum(b["evidence"]["occ_points"] for b in use)),
        "min_camera_distance": round(min(distances), 3) if distances else None,
    }


def instance_record(inst: Instance, recon: Mapping[str, Any]) -> dict[str, Any]:
    """JSON record of one instance (docs/interfaces.md)."""
    lo, hi = inst.points.min(0), inst.points.max(0)
    scores = [v.score for v in inst.views]
    return {
        "id": inst.id,
        "recording": RECORDINGS[inst.rec],
        "kind": inst.kind,
        "label": "object",
        "blob_ids": inst.blob_ids,
        "n_points": len(inst.points),
        "centroid": inst.points.mean(0).round(4).tolist(),
        "aabb": [lo.round(4).tolist(), hi.round(4).tolist()],
        "extents": (hi - lo).round(4).tolist(),
        "obb": floor_obb(inst.points),
        "views": [{"frame": v.frame, "file": str(recon["files"][v.frame]), "box": list(v.box),
                   "mask_box": list(mask_box(v.mask)), "score": round(v.score, 4)}
                  for v in inst.views],
        "mask_scores": [round(s, 4) for s in scores],
        "mask_score_mean": round(float(np.mean(scores)), 4),
        "view_consistency": None if inst.view_consistency is None
        else round(inst.view_consistency, 4),
        "feature_index": inst.id - 1,
        "n_frame_masks": len(inst.frame_masks),
        "evidence": aggregate_evidence(inst.blobs),
    }


def write_recording(
    folder: Path, rec: int, instances: Sequence[Instance], recon: Mapping[str, Any], dim: int
) -> None:
    """`<rec>.json`, `<rec>_feats.npy`, `<rec>_cls.npy`, `<rec>_points.npz`,
    `<rec>_masks.npz` of one recording (instances sorted by id)."""
    name = RECORDINGS[rec]
    height, width = recon["depth"].shape[1:]
    records = [instance_record(i, recon) for i in instances]
    (folder / f"{name}.json").write_text(json.dumps(
        {"recording": name, "image_size": [int(width), int(height)], "instances": records},
        indent=1))
    feats = np.stack([i.feature for i in instances]) if instances else np.zeros((0, dim))
    cls = np.stack([i.cls for i in instances]) if instances else np.zeros((0, dim))
    np.save(folder / f"{name}_feats.npy", feats.astype(np.float32))
    np.save(folder / f"{name}_cls.npy", cls.astype(np.float32))
    empty = [np.zeros((0, 3))]
    np.savez_compressed(
        folder / f"{name}_points.npz",
        points=np.concatenate([i.points for i in instances] or empty).astype(np.float32),
        colors=np.concatenate([i.colors for i in instances] or empty).astype(np.uint8),
        instance=np.concatenate([np.full(len(i.points), i.id) for i in instances]
                                or [np.zeros(0)]).astype(np.int32))
    entries = [(str(recon["files"][f]), i.id, s, rle["counts"])
               for i in instances for f, rle, s in i.frame_masks]
    np.savez_compressed(
        folder / f"{name}_masks.npz",
        keyframe=np.array([e[0] for e in entries], dtype=np.str_),
        instance_id=np.array([e[1] for e in entries], np.int32),
        score=np.array([e[2] for e in entries], np.float32),
        rle=np.array([e[3] for e in entries], dtype=np.str_),
        size=np.array([height, width], np.int64))


def load_instances(out_dir: str | Path, recording: str) -> dict[str, Any]:
    """Instances of one recording: `instances` (JSON records), `feats` (N, D), `cls`
    (N, D) and `points` / `colors` per instance id."""
    folder = OutputPaths(out_dir).instances
    doc = json.loads((folder / f"{recording}.json").read_text())
    with np.load(folder / f"{recording}_points.npz") as f:
        points, colors, owner = f["points"], f["colors"], f["instance"]
    return {
        "instances": doc["instances"],
        "image_size": doc["image_size"],
        "feats": np.load(folder / f"{recording}_feats.npy"),
        "cls": np.load(folder / f"{recording}_cls.npy"),
        "points": {r["id"]: points[owner == r["id"]] for r in doc["instances"]},
        "colors": {r["id"]: colors[owner == r["id"]] for r in doc["instances"]},
    }


def load_masks(path: str | Path) -> dict[str, list[tuple[int, float, dict[str, Any]]]]:
    """`<rec>_masks.npz` as {keyframe file: [(instance id, score, RLE), ...]}."""
    out: dict[str, list[tuple[int, float, dict[str, Any]]]] = {}
    with np.load(path) as f:
        size = [int(s) for s in f["size"]]
        for file, inst, score, counts in zip(f["keyframe"], f["instance_id"], f["score"],
                                             f["rle"]):
            out.setdefault(str(file), []).append(
                (int(inst), float(score), {"size": size, "counts": str(counts)}))
    return out


# ------------------------------------------------------------------ stage


def _cuda_peak_gb() -> dict[str, float] | None:
    """Peak VRAM of this process if torch has initialised CUDA, else None."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return None
    return {"allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2)}


def run_instances(
    out_dir: str | Path, cfg: Mapping[str, Any], segmenter: Segmenter | None = None,
    featurizer: Featurizer | None = None,
) -> dict[str, Any]:
    """Build the instances of both recordings from `<out>/visibility/` and write
    `<out>/instances/`. Real models are loaded unless both are given. Returns meta.json."""
    params = dict(cfg["instances"])
    paths = OutputPaths(out_dir)
    times: dict[str, float] = {}
    start = time.perf_counter()
    recon = load_recon(paths.recon / "recon.npz")
    candidates = json.loads((paths.visibility / "candidates.json").read_text())["candidates"]
    grid = load_grid(paths.visibility / "grid.npz")
    times["load"] = time.perf_counter() - start

    t0 = time.perf_counter()
    if segmenter is None or featurizer is None:
        from roomdiff.models import load_models

        segmenter, featurizer = load_models(params)
    times["load_models"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    selected, dropped = select_blobs(candidates, params["min_unverified_volume"],
                                     params["hidden_surface_margin"],
                                     params["max_unverified_extent"])
    frames = [np.flatnonzero(recon["rec"] == r) for r in range(len(RECORDINGS))]
    static = [static_surface_test(grid, 1 - r) if params["drop_static"] else None
              for r in range(len(RECORDINGS))]
    raw: list[list[Instance]] = [[] for _ in RECORDINGS]
    for blob in selected:
        own = CANDIDATE_TYPES[blob["type"]][0]
        voxels = blob_voxel_centres(grid, blob)
        inst, reason = build_instance(blob, voxels, recon, frames[own], segmenter, params,
                                      static[own])
        if inst is None:
            dropped.append({"blob": blob["id"], "reason": reason})
        else:
            raw[own].append(inst)
    times["segment_and_lift"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    final: list[list[Instance]] = []
    for rec in range(len(RECORDINGS)):
        merged = merge_instances(raw[rec], params["merge_mask_iou"], params["merge_point_dist"],
                                 params["merge_point_frac"], params["lift_voxel"])
        merged.sort(key=lambda i: (i.kind == "unverified", -len(i.points)))
        for k, inst in enumerate(merged, start=1):
            inst.id = k
        final.append(merged)
    times["merge"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for instances in final:
        for inst in instances:
            describe_features(inst, recon, featurizer, params["mask_erode_px"],
                              params["cls_box_enlarge"])
    times["features"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for rec, instances in enumerate(final):
        for inst in instances:
            inst.frame_masks = frame_masks(
                inst, frames[rec], recon, segmenter, params["frame_visible_frac"],
                params["frame_depth_tol"], params["frame_prompt_points"], params["box_enlarge"])
    times["frame_masks"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    folder = paths.instances
    folder.mkdir(parents=True, exist_ok=True)
    for rec, instances in enumerate(final):
        write_recording(folder, rec, instances, recon, featurizer.dim)
    times["write"] = time.perf_counter() - t0
    times["total"] = time.perf_counter() - start

    counts = {name: len(final[r]) for r, name in enumerate(RECORDINGS)}
    meta = {
        "params": params,
        "blobs_selected": [b["id"] for b in selected],
        "blobs_dropped": dropped,
        "fragments": {name: len(raw[r]) for r, name in enumerate(RECORDINGS)},
        "instances": counts,
        "kinds": {name: {k: sum(i.kind == k for i in final[r])
                         for k in (*VERIFIED_TYPES, "unverified")}
                  for r, name in enumerate(RECORDINGS)},
        "frame_masks": {name: sum(len(i.frame_masks) for i in final[r])
                        for r, name in enumerate(RECORDINGS)},
        "runtime_s": {k: round(v, 2) for k, v in times.items()},
        "peak_vram_gb": _cuda_peak_gb(),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[instances] {len(selected)} blobs -> fragments "
          f"{meta['fragments']['before']} before / {meta['fragments']['after']} after -> "
          f"instances {counts['before']} / {counts['after']}, {times['total']:.1f} s -> {folder}")
    return meta
