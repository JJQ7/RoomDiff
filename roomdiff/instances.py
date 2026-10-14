"""M4 instances: object instances from the M3 change blobs, per recording.

Each selected blob (all disappeared / appeared blobs and the large unverified ones) is
handled in its *owner* recording (where it is occupied):

1. keyframes that see the blob well are chosen (`roomdiff.projection.score_views` on its
   voxel centres: large, depth-consistent and uncut in the image; varied viewpoints);
2. SAM 2.1 is prompted there with the box of the projected voxel centres and the
   depth-visible ones; the best mask that stays inside the box is kept; a mask that
   reaches a side of the box is re-prompted with the box extended on that side (the
   object continues beyond the blob), as long as the new mask contains the old one;
3. all valid-depth mask pixels are lifted to 3D; the connected component(s) holding the
   blob are kept, minus static surfaces shared with the other recording (a mask that
   grew far beyond the blob falls back to the blob's neighbourhood, `lift_fallback`);
4. fragments of one object are merged (shared-view mask IoU or 3D overlap), then
   touching fragments of the same kind (a chair seat and its base) if the union stays
   object-sized (`merge_close_fragments`);
5. the appearance clusters of `<out>/appearance/` (`roomdiff.appearance`: small or flat
   changes the voxel differencing misses) that no instance covers yet are appended
   (`source: "appearance"`, kind disappeared / appeared by recording);
6. each instance gets a DINOv3 feature (mask-pooled patch tokens, averaged over views)
   and a CLS token of its box crop, and SAM masks in every keyframe of its recording
   where its points are visible (COCO RLE, for the SceneDiff evaluation); SAM sees the
   full-resolution keyframe JPEG there, and each mask is kept at that resolution
   (`rle_full`) and downsampled to the reconstruction resolution (`rle`).

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
from roomdiff.projection import MIN_VISIBLE_FRAC, project_to_pixels, score_views, subsample
from roomdiff.recon import load_recon
from roomdiff.visibility import CANDIDATE_TYPES, OCCUPIED, load_grid

VERIFIED_TYPES = ("disappeared", "appeared")
MIN_MASK_IN_BOX = 0.5  # a SAM mask must have at least this fraction of its pixels in the box
MAX_PROJECTED = 2000  # points projected per view test (random subset)
OUTLIER_RADIUS = 0.03  # m; lifted points with fewer than OUTLIER_MIN_NEIGHBOURS other ...
OUTLIER_MIN_NEIGHBOURS = 3  # ... points within this radius are dropped
SEED_SHARE = 0.1  # lifting keeps the components holding at least this share of the blob voxels
GROW_EDGE_PX = 2  # a mask within this many px of a prompt box side reaches it ...
GROW_TOUCH_SPAN = 0.25  # ... if those pixels span this share of the mask's extent along the side
GROW_KEEP = 0.8  # a re-prompted (grown) mask must contain this share of the previous one
KEYFRAME_ASPECT_TOL = 0.1  # max aspect difference keyframe JPEG / recon image (frame_masks)

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
    # per-keyframe masks (`frame_masks`): frame, RLE at the recon resolution, SAM score,
    # RLE at the keyframe resolution (None where SAM saw the recon image)
    frame_masks: list[tuple[int, dict[str, Any], float, dict[str, Any] | None]] = field(
        default_factory=list)
    merged_from: list[list[int]] = field(default_factory=list)  # blob ids per gap-merged part
    lift_fallback: bool = False  # some view's mask grew too far: bounded lifting was used
    lift_growth: list[float | None] = field(default_factory=list)  # per view (`growth`)
    source: str = "geometry"  # "geometry" (M3 blobs) | "appearance" (roomdiff.appearance)
    evidence: dict[str, Any] | None = None  # set for appearance instances (no blobs)

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
    depth_tol: float, min_angle_deg: float, on_surface: bool = True,
) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Up to `n_views` keyframes `(frame, box)` that show the object `points` well; `box`
    = 2D box of the points projecting inside the image.

    Views are ranked by `roomdiff.projection.score_views` (box area of the visible
    points x visible fraction x border / frontal factors; visible = depth-consistent
    within `depth_tol`, or with `on_surface` False not occluded) and need at least
    `MIN_VISIBLE_FRAC` of the points visible; they are then picked greedily so that the
    viewing directions differ by at least `min_angle_deg`, remaining slots by rank. All of
    it is independent of the world frame.
    """
    if len(points) == 0 or len(frames) == 0:
        return []
    ranked = [v for v in score_views(points, frames, recon["cam2world"], recon["K"],
                                     recon["depth"], recon["valid"], depth_tol, on_surface)
              if v.visible >= MIN_VISIBLE_FRAC]
    centroid = np.asarray(points, np.float64).mean(0)
    directions = {}
    for v in ranked:
        d = recon["cam2world"][v.frame][:3, 3] - centroid
        directions[v.frame] = d / max(np.linalg.norm(d), 1e-9)
    min_cos = np.cos(np.radians(min_angle_deg))
    chosen: list[tuple[int, tuple[int, int, int, int]]] = []
    for v in ranked:
        if len(chosen) < n_views and all(directions[v.frame] @ directions[f] < min_cos
                                         for f, _ in chosen):
            chosen.append((v.frame, v.box))
    for v in ranked:
        if len(chosen) < n_views and all(v.frame != f for f, _ in chosen):
            chosen.append((v.frame, v.box))
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


def pick_index(
    masks: Sequence[np.ndarray], scores: Sequence[float], box: Sequence[int],
    contain: np.ndarray | None = None,
) -> int | None:
    """Index of the highest-scoring non-empty mask with at least `MIN_MASK_IN_BOX` of its
    pixels inside `box` (and, if given, `GROW_KEEP` of the pixels of the mask `contain`),
    or None."""
    x0, y0, x1, y1 = box
    best = None
    for i, (mask, score) in enumerate(zip(masks, scores)):
        area = int(mask.sum())
        if area == 0 or mask[y0:y1 + 1, x0:x1 + 1].sum() < MIN_MASK_IN_BOX * area:
            continue
        if contain is not None and (mask & contain).sum() < GROW_KEEP * contain.sum():
            continue
        if best is None or score > scores[best]:
            best = i
    return best


def pick_mask(
    masks: np.ndarray, scores: np.ndarray, box: Sequence[int],
    contain: np.ndarray | None = None,
) -> tuple[np.ndarray, float] | None:
    """Highest-scoring non-empty mask with at least `MIN_MASK_IN_BOX` of its pixels inside
    `box` (and, if given, `GROW_KEEP` of the pixels of the mask `contain`), or None."""
    i = pick_index(masks, scores, box, contain)
    return None if i is None else (masks[i], float(scores[i]))


def touched_sides(
    mask: np.ndarray, box: Sequence[int], width: int, height: int
) -> tuple[bool, bool, bool, bool]:
    """(left, top, right, bottom): the mask reaches that side of `box` (pixels within
    `GROW_EDGE_PX`, spanning at least `GROW_TOUCH_SPAN` of the mask's extent along it) and
    the side is not on the image border, so the object may continue beyond the box."""
    x0, y0, x1, y1 = box
    rows, cols = np.nonzero(mask[y0:y1 + 1, x0:x1 + 1])
    if len(rows) == 0:
        return False, False, False, False
    h, w = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1

    def reaches(values: np.ndarray, extent: int) -> bool:
        return bool(len(values)) and values.max() - values.min() + 1 >= GROW_TOUCH_SPAN * extent

    return (x0 > 0 and reaches(rows[cols <= GROW_EDGE_PX], h),
            y0 > 0 and reaches(cols[rows <= GROW_EDGE_PX], w),
            x1 < width - 1 and reaches(rows[cols >= x1 - x0 - GROW_EDGE_PX], h),
            y1 < height - 1 and reaches(cols[rows >= y1 - y0 - GROW_EDGE_PX], w))


def grow_box(
    box: Sequence[int], sides: Sequence[bool], width: int, height: int
) -> tuple[int, int, int, int]:
    """`box` extended by its own width / height on each side in `sides` (left, top,
    right, bottom), clipped to the image."""
    x0, y0, x1, y1 = (int(v) for v in box)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    left, top, right, bottom = sides
    return (max(0, x0 - bw) if left else x0, max(0, y0 - bh) if top else y0,
            min(width - 1, x1 + bw) if right else x1, min(height - 1, y1 + bh) if bottom else y1)


def segment_view(
    segmenter: Segmenter, recon: Mapping[str, Any], frame: int, voxels: np.ndarray,
    box: Sequence[int], box_enlarge: float, n_points: int, depth_tol: float,
    grow_rounds: int = 0,
) -> View | None:
    """SAM mask of the blob in one keyframe: prompt = projected box enlarged by
    `box_enlarge` plus up to `n_points` projected voxel centres lying on the observed
    surface (|observed depth - depth| <= `depth_tol`). Up to `grow_rounds` times, a mask
    that reaches a side of the prompt box (`touched_sides`) is re-prompted with the box
    grown on those sides (`grow_box`); the new mask must contain the old one."""
    height, width = recon["depth"].shape[1:]
    prompt_box = enlarge_box(box, box_enlarge, width, height)
    col, row, z, _, depth, valid = project_with_depth(voxels, frame, recon)
    on_surface = valid & (np.abs(depth - z) <= depth_tol)
    points = prompt_points(col[on_surface], row[on_surface], n_points) if on_surface.any() \
        else None
    image = recon["images"][frame]
    picked = pick_mask(*segmenter.segment(str(frame), image, prompt_box, points), prompt_box)
    if picked is None:
        return None
    mask, score = picked
    for _ in range(grow_rounds):
        sides = touched_sides(mask, prompt_box, width, height)
        if not any(sides):
            break
        bigger = grow_box(prompt_box, sides, width, height)
        grown = pick_mask(*segmenter.segment(str(frame), image, bigger, points), bigger, mask)
        if grown is None:
            break
        prompt_box, (mask, score) = bigger, grown
    return View(int(frame), prompt_box, mask, score)


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


def connected_to_blob(
    points: np.ndarray, voxels: np.ndarray, seed_radius: float, cell: float, link: float
) -> np.ndarray | None:
    """Which `points` lie in a 3D connected component that holds the blob. Points are
    binned into cubic cells of `cell` m; two cells are connected when their centres are
    at most `link` m apart. Each blob voxel centre within `seed_radius` of a cell votes for
    that cell's component; components with at least `SEED_SHARE` of the votes are kept.
    None if no blob voxel is near any point (the mask missed the blob)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if len(points) == 0 or len(voxels) == 0:
        return None
    ijk = np.floor(np.asarray(points, np.float64) / cell).astype(np.int64)
    cells, owner = np.unique(ijk, axis=0, return_inverse=True)
    owner = owner.ravel()
    tree = cKDTree((cells + 0.5) * cell)
    pairs = tree.query_pairs(link + 1e-9, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(cells), len(cells)))
    _, comp = connected_components(graph, directed=False)
    dist, nearest = tree.query(voxels, distance_upper_bound=seed_radius)
    hit = comp[nearest[np.isfinite(dist)]]
    if len(hit) == 0:
        return None
    votes = np.bincount(hit, minlength=comp.max() + 1)
    keep = votes >= max(1.0, SEED_SHARE * len(hit))
    return keep[comp[owner]]


def growth(points: np.ndarray, voxels: np.ndarray, blob_voxel: float) -> float:
    """Volume of the oriented box (`floor_obb`) of `points` over that of the blob: the
    box of its voxel centres grown by one voxel on every side (M3 trims blobs where they
    come within `dist_margin` of the other recording's surfaces, so they fall short of
    the object by a voxel or two)."""
    blob = np.prod(np.asarray(floor_obb(voxels)["extents"]) + 2 * blob_voxel)
    return float(np.prod(floor_obb(points)["extents"]) / max(blob, 1e-12))


def lift_view_points(
    points: np.ndarray, voxels: np.ndarray, params: Mapping[str, Any],
    on_static: PointTest | None = None, blob_voxel: float = 0.02,
) -> tuple[np.ndarray, bool, float | None]:
    """Which lifted mask points of one view belong to the object, whether the bounded
    fallback was used, and the growth of the connected part (None if there was none).

    Keeps the connected components that hold the blob (`connected_to_blob`, cells of
    `lift_cell_m`, link `lift_link_m`), then drops points on static surfaces
    (`on_static`) unless they lie within `near_voxel_m` of a blob voxel. If that leaves
    nothing, misses the blob, or the outlier-filtered rest spans an oriented box of more
    than `max_growth` x the blob's (the mask bled into a wall or the floor that the other
    recording did not see), the old bounded rule `keep_near_blob` is used instead."""
    seed_radius = 0.5 * np.sqrt(3.0) * blob_voxel + params["lift_cell_m"]
    keep = connected_to_blob(points, voxels, seed_radius, params["lift_cell_m"],
                             params["lift_link_m"])
    if keep is not None and on_static is not None and keep.any():
        idx = np.flatnonzero(keep)
        static = idx[on_static(points[idx])]
        if len(static):
            dist, _ = cKDTree(voxels).query(points[static],
                                            distance_upper_bound=params["near_voxel_m"])
            keep[static[~np.isfinite(dist)]] = False
    ratio = None
    if keep is not None and keep.any():
        kept, _ = clean_points(points[keep], np.zeros((int(keep.sum()), 3), np.uint8),
                               params["lift_voxel"])
        ratio = growth(kept, voxels, blob_voxel) if len(kept) else None
        if ratio is not None and ratio <= params["max_growth"]:
            return keep, False, ratio
    return keep_near_blob(points, voxels, params["expand_m"], params["near_voxel_m"],
                          on_static), True, ratio


def build_instance(
    blob: Mapping[str, Any], voxels: np.ndarray, recon: Mapping[str, Any],
    frames: Sequence[int], segmenter: Segmenter, params: Mapping[str, Any],
    on_static: PointTest | None = None, blob_voxel: float = 0.02,
) -> tuple[Instance | None, str]:
    """Instance of one blob (views, masks, lifted points) or None with the reason.
    `blob_voxel`: voxel size of the M3 grid."""
    own = CANDIDATE_TYPES[blob["type"]][0]
    kind = blob["type"] if blob["type"] in VERIFIED_TYPES else "unverified"
    chosen = select_views(voxels, frames, recon, params["n_views"], params["occlusion_tol"],
                          params["min_view_angle"])
    if not chosen:
        return None, "no clear view"
    views, points, colors, fallback, growths = [], [], [], False, []
    for frame, box in chosen:
        view = segment_view(segmenter, recon, frame, voxels, box, params["box_enlarge"],
                            params["n_prompt_points"], params["occlusion_tol"],
                            params["mask_grow_rounds"])
        if view is None:
            continue
        p, c = lift_mask(recon, frame, view.mask)
        keep, fell_back, ratio = lift_view_points(p, voxels, params, on_static, blob_voxel)
        fallback |= fell_back
        growths.append(None if ratio is None else round(ratio, 2))
        views.append(view)
        points.append(p[keep])
        colors.append(c[keep])
    if not views:
        return None, "no SAM mask inside the box"
    pts, cols = clean_points(np.concatenate(points), np.concatenate(colors),
                             params["lift_voxel"])
    if len(pts) < params["min_points"]:
        return None, f"{len(pts)} points < min_points"
    inst = Instance(own, kind, [dict(blob)], voxels, pts, cols, views)
    inst.lift_fallback, inst.lift_growth = fallback, growths
    return inst, "ok"


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
    merged = Instance(group[0].rec, kind, [b for i in group for b in i.blobs],
                      np.concatenate([i.voxels for i in group]), points, colors,
                      sorted(views.values(), key=lambda v: -v.score))
    merged.lift_fallback = any(i.lift_fallback for i in group)
    return merged


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


def near_fraction(a: np.ndarray, b: np.ndarray, gap: float) -> float:
    """Fraction of the points of the smaller set lying within `gap` of the other set."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    return float(np.isfinite(cKDTree(big).query(small, distance_upper_bound=gap)[0]).mean())


def merge_close_fragments(
    instances: Sequence[Instance], gap: float, frac: float, max_extent: float, voxel: float
) -> tuple[list[Instance], list[dict[str, Any]]]:
    """Join fragments of one object that touch without overlapping (a chair seat and its
    base, found by two blobs): two instances of the same recording and kind are merged
    when at least `frac` of the points of the smaller one lie within `gap` m of the other's
    points and every extent of the union's oriented box (`floor_obb`) is below
    `max_extent`. Greedy: the pair with the largest near fraction first, re-evaluated after
    every merge. The merged instance lists the blob ids of its parts in `merged_from`.
    Returns the instances and one record per merge."""
    current = list(instances)
    records: list[dict[str, Any]] = []
    while True:
        best = None
        for i, a in enumerate(current):
            for j in range(i + 1, len(current)):
                b = current[j]
                if a.rec != b.rec or a.kind != b.kind or not len(a.points) or not len(b.points):
                    continue
                if np.any(a.points.min(0) - gap > b.points.max(0)) or \
                        np.any(b.points.min(0) - gap > a.points.max(0)):
                    continue  # boxes farther apart than the gap
                near = near_fraction(a.points, b.points, gap)
                if near < frac or (best is not None and near <= best[0]):
                    continue
                extents = floor_obb(np.vstack([a.points, b.points]))["extents"]
                if max(extents) < max_extent:
                    best = (near, i, j, extents)
        if best is None:
            return current, records
        near, i, j, extents = best
        a, b = current[i], current[j]
        merged = combine([a, b], voxel)
        merged.merged_from = (a.merged_from or [a.blob_ids]) + (b.merged_from or [b.blob_ids])
        records.append({"recording": RECORDINGS[a.rec], "kind": a.kind,
                        "parts": [a.blob_ids, b.blob_ids], "near_fraction": round(near, 4),
                        "union_obb_extents": extents})
        current = [x for k, x in enumerate(current) if k not in (i, j)] + [merged]


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


def downsample_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bool mask resized to (height, width) by area averaging: a pixel is set when more
    than half of its area is."""
    import cv2

    if mask.shape == (height, width):
        return mask.astype(bool)
    resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    return resized > 0.5


def load_keyframe(path: str | Path) -> np.ndarray | None:
    """RGB uint8 image (H, W, 3) of a keyframe JPEG, None if the file does not exist."""
    from PIL import Image

    if not Path(path).is_file():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def full_keyframe(
    frame: int, recon: Mapping[str, Any], keyframes: Path, cache: dict[int, np.ndarray | None]
) -> np.ndarray | None:
    """Full-resolution image of keyframe `frame` (`keyframes / recon["files"][frame]`),
    read once into `cache`; None if the file is missing or its aspect ratio differs from
    the recon image's by more than `KEYFRAME_ASPECT_TOL` (another crop or orientation)."""
    if frame not in cache:
        image = load_keyframe(keyframes / str(recon["files"][frame]))
        height, width = recon["depth"].shape[1:]
        if image is not None and \
                abs(image.shape[1] * height / (image.shape[0] * width) - 1) > KEYFRAME_ASPECT_TOL:
            image = None
        cache[frame] = image
    return cache[frame]


def frame_masks(
    inst: Instance, frames: Sequence[int], recon: Mapping[str, Any], segmenter: Segmenter,
    visible_frac: float, depth_tol: float, n_points: int, box_enlarge: float,
    keyframes: str | Path | None = None,
    full_images: dict[int, np.ndarray | None] | None = None,
) -> list[tuple[int, dict[str, Any], float, dict[str, Any] | None]]:
    """SAM masks `(frame, RLE, score, full RLE)` of the instance in every keyframe where at
    least `visible_frac` of its points are depth-consistent (|observed - projected depth|
    <= `depth_tol`); prompt = up to `n_points` of those pixels plus their box.

    With `keyframes` (the recording's keyframe folder, file names `recon["files"]`) SAM is
    prompted on the full-resolution keyframe JPEG, box and points scaled from recon pixels
    by the width / height ratios (embedding key `full:<frame>`, never the recon image's
    `<frame>`): the full RLE is its mask, the RLE that mask downsampled to the recon
    resolution (`downsample_mask`; the mask is picked on it, as before), the score that of
    this run. JPEGs are read once into `full_images` (one dict shared by the instances of a
    recording). Without a usable JPEG (`full_keyframe`) the recon image is prompted as
    before and the full RLE is None."""
    height, width = recon["depth"].shape[1:]
    full_images = {} if full_images is None else full_images
    sample = subsample(inst.points, MAX_PROJECTED)
    out: list[tuple[int, dict[str, Any], float, dict[str, Any] | None]] = []
    for frame in frames:
        col, row, z, _, depth, valid = project_with_depth(sample, frame, recon)
        consistent = valid & (np.abs(depth - z) <= depth_tol)
        if consistent.mean() < visible_frac:
            continue
        c, r = col[consistent], row[consistent]
        box = enlarge_box((c.min(), r.min(), c.max(), r.max()), box_enlarge, width, height)
        points = prompt_points(c, r, n_points)
        full = None if keyframes is None else \
            full_keyframe(int(frame), recon, Path(keyframes), full_images)
        if full is None:
            masks, scores = segmenter.segment(str(frame), recon["images"][frame], box, points)
            picked = pick_mask(masks, scores, box)
            if picked is not None:
                out.append((int(frame), encode_rle(picked[0]), picked[1], None))
            continue
        sx, sy = full.shape[1] / width, full.shape[0] / height
        big, scores = segmenter.segment(f"full:{frame}", full,
                                        (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy),
                                        points * np.array([sx, sy]))
        small = [downsample_mask(m, height, width) for m in big]
        i = pick_index(small, scores, box)
        if i is not None:
            out.append((int(frame), encode_rle(small[i]), float(scores[i]), encode_rle(big[i])))
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
        "merged_from": inst.merged_from,
        "lift_fallback": inst.lift_fallback,
        "source": inst.source,
        # M3 demoted a source blob to unverified (`demotion_reason`, e.g. "weak_alignment")
        "demoted": next((b["demotion_reason"] for b in inst.blobs
                         if b.get("demotion_reason")), None),
        "evidence": inst.evidence if inst.evidence is not None
        else aggregate_evidence(inst.blobs),
    }


def write_recording(
    folder: Path, rec: int, instances: Sequence[Instance], recon: Mapping[str, Any], dim: int
) -> None:
    """`<rec>.json`, `<rec>_feats.npy`, `<rec>_cls.npy`, `<rec>_points.npz`,
    `<rec>_masks.npz` of one recording (instances sorted by id)."""
    name = RECORDINGS[rec]
    height, width = recon["depth"].shape[1:]
    records = [instance_record(i, recon) for i in instances]
    full = next((rle["size"] for i in instances for *_, rle in i.frame_masks
                 if rle is not None), None)  # [H, W] of the keyframe-resolution masks
    (folder / f"{name}.json").write_text(json.dumps(
        {"recording": name, "image_size": [int(width), int(height)],
         "image_size_full": None if full is None else [int(full[1]), int(full[0])],
         "instances": records}, indent=1))
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
    save_masks(folder / f"{name}_masks.npz", instances, recon["files"], height, width)


def save_masks(
    path: str | Path, instances: Sequence[Instance], files: Sequence[str], height: int,
    width: int,
) -> None:
    """`<rec>_masks.npz`: one row per per-keyframe mask (`Instance.frame_masks`) of the
    instances: `keyframe` (file name `files[frame]`), `instance_id`, SAM `score`, `rle`
    (COCO RLE counts) at the recon resolution `size` [height, width], `rle_full` (counts at
    the keyframe resolution, "" where SAM saw the recon image) and `size_full` ([H, W] of
    each `rle_full`, [0, 0] without)."""
    entries = [(str(files[f]), i.id, s, rle["counts"], full)
               for i in instances for f, rle, s, full in i.frame_masks]
    np.savez_compressed(
        path,
        keyframe=np.array([e[0] for e in entries], dtype=np.str_),
        instance_id=np.array([e[1] for e in entries], np.int32),
        score=np.array([e[2] for e in entries], np.float32),
        rle=np.array([e[3] for e in entries], dtype=np.str_),
        size=np.array([height, width], np.int64),
        rle_full=np.array([e[4]["counts"] if e[4] else "" for e in entries], dtype=np.str_),
        size_full=np.array([e[4]["size"] if e[4] else [0, 0] for e in entries],
                           np.int64).reshape(-1, 2))


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


def load_masks(
    path: str | Path, full: bool = False
) -> dict[str, list[tuple[int, float, dict[str, Any]]]]:
    """`<rec>_masks.npz` as {keyframe file: [(instance id, score, RLE), ...]}: RLEs at the
    recon resolution, or with `full` the keyframe-resolution ones (`rle_full`) where there
    are any (else the recon RLE, e.g. in files written before `rle_full` existed)."""
    out: dict[str, list[tuple[int, float, dict[str, Any]]]] = {}
    with np.load(path) as f:
        size = [int(s) for s in f["size"]]
        n = len(f["keyframe"])
        use_full = full and "rle_full" in f.files
        counts_full = f["rle_full"] if use_full else np.full(n, "")
        sizes_full = f["size_full"] if use_full else np.zeros((n, 2), np.int64)
        for file, inst, score, counts, big_counts, big_size in zip(
                f["keyframe"], f["instance_id"], f["score"], f["rle"], counts_full, sizes_full):
            rle = {"size": [int(s) for s in big_size], "counts": str(big_counts)} \
                if str(big_counts) else {"size": size, "counts": str(counts)}
            out.setdefault(str(file), []).append((int(inst), float(score), rle))
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
                                      static[own], grid["voxel_size"])
        if inst is None:
            dropped.append({"blob": blob["id"], "reason": reason})
        else:
            raw[own].append(inst)
    times["segment_and_lift"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    final: list[list[Instance]] = []
    gap_merges: list[dict[str, Any]] = []
    for rec in range(len(RECORDINGS)):
        merged = merge_instances(raw[rec], params["merge_mask_iou"], params["merge_point_dist"],
                                 params["merge_point_frac"], params["lift_voxel"])
        merged, joined = merge_close_fragments(merged, params["merge_gap_m"],
                                               params["merge_gap_frac"],
                                               params["merge_max_extent_m"], params["lift_voxel"])
        gap_merges += joined
        merged.sort(key=lambda i: (i.kind == "unverified", -len(i.points)))
        for k, inst in enumerate(merged, start=1):
            inst.id = k
        final.append(merged)
    times["merge"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    appearance_meta = None
    app = cfg.get("appearance") or {}
    if app.get("enabled") and (paths.appearance / "clusters.json").is_file():
        from roomdiff.appearance import load_appearance_instances

        extra, appearance_meta = load_appearance_instances(
            out_dir, final, app["covered_dist"], app["covered_frac"])
        for rec, instances in enumerate(extra):  # appended after the geometric ones
            for k, inst in enumerate(instances, start=len(final[rec]) + 1):
                inst.id = k
            final[rec] += instances
    times["appearance"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for instances in final:
        for inst in instances:
            describe_features(inst, recon, featurizer, params["mask_erode_px"],
                              params["cls_box_enlarge"])
    times["features"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for rec, instances in enumerate(final):
        full_images: dict[int, np.ndarray | None] = {}  # keyframe JPEGs, read once
        for inst in instances:
            inst.frame_masks = frame_masks(
                inst, frames[rec], recon, segmenter, params["frame_visible_frac"],
                params["frame_depth_tol"], params["frame_prompt_points"], params["box_enlarge"],
                paths.keyframes_of(RECORDINGS[rec]), full_images)
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
        "lift_fallbacks": {name: [b["id"] for i in raw[r] if i.lift_fallback for b in i.blobs]
                           for r, name in enumerate(RECORDINGS)},
        "lift_growth": {str(i.blobs[0]["id"]): i.lift_growth for r in raw for i in r},
        "gap_merges": gap_merges,
        "appearance": appearance_meta,
        "instances": counts,
        "kinds": {name: {k: sum(i.kind == k for i in final[r])
                         for k in (*VERIFIED_TYPES, "unverified")}
                  for r, name in enumerate(RECORDINGS)},
        "sources": {name: {k: sum(i.source == k for i in final[r])
                           for k in ("geometry", "appearance")}
                    for r, name in enumerate(RECORDINGS)},
        "frame_masks": {name: sum(len(i.frame_masks) for i in final[r])
                        for r, name in enumerate(RECORDINGS)},
        "runtime_s": {k: round(v, 2) for k, v in times.items()},
        "peak_vram_gb": _cuda_peak_gb(),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    added = {name: meta["sources"][name]["appearance"] for name in RECORDINGS}
    extra = (f" (+ appearance {added['before']} / {added['after']})"
             if appearance_meta is not None else "")
    print(f"[instances] {len(selected)} blobs -> fragments "
          f"{meta['fragments']['before']} before / {meta['fragments']['after']} after -> "
          f"instances {counts['before']} / {counts['after']}{extra}, {times['total']:.1f} s"
          f" -> {folder}")
    return meta
