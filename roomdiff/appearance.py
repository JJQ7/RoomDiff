"""Appearance channel: changes too small or too flat for the voxel differencing (M3).

M3 needs an object to stand more than `visibility.dist_margin` (4 cm) off the surfaces of
the other recording, so a wallet on a table or a jumper on a bed is invisible to it. This
stage compares what co-visible keyframes of the two recordings *look like* at the same 3D
points (reprojected DINO feature dissimilarity, the strongest single cue of SceneDiff),
plus an asymmetric depth cue:

1. Co-visible pairs: for keyframe i of one recording and j of the other, the share of a
   stride-`covis_stride` subsample of i's valid pixels that lands inside j on a valid,
   depth-consistent pixel (|z - depth_j| < max(covis_depth_abs, covis_depth_rel z));
   score = the smaller of both directions. Every keyframe keeps its `pairs_per_frame`
   best partners with a score of at least `min_covisibility`.
2. DINOv3 ViT-L/16 patch tokens of every keyframe (24 x 42 at 384 x 672), L2-normalised,
   cached as float16 in `feats_<rec>.npy`.
3. Per pair i -> j: every patch centre of i with valid depth is reprojected into j
   (u, v, z). It is *observed* by j if it lands inside j on a valid pixel and no surface
   of j lies more than the tolerance in front of it; then `feat` = 1 - cos(F_i(p),
   F_j(u, v)) (bilinear in j's token grid) and `geom` = 1 if depth_j(u, v) - z >
   `geom_tol_abs + geom_tol_rel z` (j sees through the place: the surface seen in i is
   gone). A point behind j's surface (occluded, or something appeared in j in front of
   it) or outside j gets no vote in this direction; what appeared is found from j's side.
4. Per keyframe: score = mean over the partners that observed the patch of
   (geom + `w_feat` feat), 0 with fewer than `min_votes` such partners; bilinearly
   upsampled to the image, thresholded at `score_threshold` (`threshold_mode: fixed`) or
   at the max-entropy (Kapur) threshold of all scores (`max_entropy`; always reported);
   connected regions of `min_region_frac` .. `max_region_frac` of the image (at most
   `max_regions_per_frame`, highest mean score first).
5. Per region: SAM 2.1 with the region's box (enlarged by `box_enlarge`) and up to
   `n_prompt_points` highest-scoring patch centres inside it; the best-scoring mask whose
   IoU with the region exceeds `min_mask_iou` is lifted to 3D (valid-depth pixels, 1 cm
   voxels, outliers removed). A candidate whose score came mainly from the depth cue
   (feature share <= 0.5) and whose points lie mostly (> `static_frac`) within
   `static_dist` of the other recording's surfaces is dropped (a static surface);
   feature-driven candidates are kept (an object lying flat on a surface, or replaced in
   place). Candidates of one recording are clustered across keyframes (more than
   `cluster_frac` of the points of one within `cluster_dist` of the other); clusters
   supported by at least `min_frames` keyframes are written. A cluster keeps the points
   that at least `CONSENSUS_SHARE` of its keyframes able to see them (depth-consistent)
   put inside their mask, so a mask that bled onto a neighbour in a few views does not
   carry it along.

The instance stage (`roomdiff.instances.run_instances`) appends the clusters that no M4
instance covers yet as instances with `source: "appearance"`
(`load_appearance_instances`). Writes `<out>/appearance/` (docs/interfaces.md).
"""

from __future__ import annotations

import json
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from roomdiff.geometry import (
    floor_obb,
    project_points,
    se3_inverse,
    unproject_depth,
    voxel_downsample,
)
from roomdiff.instances import (
    Instance,
    View,
    clean_points,
    decode_rle,
    encode_rle,
    enlarge_box,
    lift_mask,
    mask_box,
    mask_iou,
    point_overlap,
)
from roomdiff.models import Featurizer, Segmenter
from roomdiff.paths import RECORDINGS, OutputPaths
from roomdiff.projection import MIN_VISIBLE_FRAC, project_to_pixels, score_views
from roomdiff.recon import load_recon

KINDS = ("disappeared", "appeared")  # instance kind of a cluster owned by before / after
FEATURE_DRIVEN = 0.5  # feature term > this share of the score: kept on static surfaces
STATIC_VOXEL = 0.01  # m, voxel size of the other recording's cloud for the static test
PLACE_DEPTH_TOL = 0.05  # m, place-observing views: not occluded by more than this
CONSENSUS_SHARE = 0.5  # cluster points: inside the masks of this share of the views seeing them
CONSENSUS_MIN_FRAMES = 3  # ... voted on only with at least this many keyframes


# ------------------------------------------------------------------ co-visibility


def depth_tolerance(z: np.ndarray, rel: float, abs_: float) -> np.ndarray:
    """Depth-consistency tolerance max(abs_, rel z) (metres)."""
    return np.maximum(abs_, rel * np.asarray(z, np.float64))


def stride_points(recon: Mapping[str, Any], i: int, stride: int) -> np.ndarray:
    """World points (M, 3) of every `stride`-th valid pixel (rows and columns) of keyframe i."""
    depth = recon["depth"][i]
    sel = np.zeros(depth.shape, bool)
    sel[::stride, ::stride] = True
    sel &= recon["valid"][i] & (depth > 0)
    return unproject_depth(depth, recon["K"][i], recon["cam2world"][i])[sel]


def consistent_share(points: np.ndarray, recon: Mapping[str, Any], j: int, rel: float,
                     abs_: float) -> float:
    """Share of `points` that project inside keyframe j onto a valid pixel whose depth is
    within the tolerance of their own (0 for no points)."""
    if len(points) == 0:
        return 0.0
    height, width = recon["depth"].shape[1:]
    col, row, z, inside = project_to_pixels(points, recon["K"][j], recon["cam2world"][j],
                                            width, height)
    idx = np.flatnonzero(inside)
    d = recon["depth"][j][row[idx], col[idx]]
    ok = recon["valid"][j][row[idx], col[idx]] & (
        np.abs(z[idx] - d) < depth_tolerance(z[idx], rel, abs_))
    return float(ok.sum() / len(points))


def covisibility(
    recon: Mapping[str, Any], frames_a: Sequence[int], frames_b: Sequence[int], stride: int,
    rel: float, abs_: float,
) -> np.ndarray:
    """Symmetric co-visibility (len(a), len(b)): min of the two directional shares."""
    points = {int(i): stride_points(recon, i, stride) for i in [*frames_a, *frames_b]}
    out = np.zeros((len(frames_a), len(frames_b)))
    for k, i in enumerate(frames_a):
        for m, j in enumerate(frames_b):
            ab = consistent_share(points[int(i)], recon, j, rel, abs_)
            out[k, m] = 0.0 if ab == 0 else min(ab, consistent_share(points[int(j)], recon, i,
                                                                     rel, abs_))
    return out


def select_partners(scores: np.ndarray, k: int, min_score: float) -> list[list[int]]:
    """Per row: the columns of the `k` best scores of at least `min_score`, best first."""
    out = []
    for row in np.asarray(scores):
        order = np.argsort(-row, kind="stable")[:k]
        out.append([int(m) for m in order if row[m] >= min_score])
    return out


# ------------------------------------------------------------------ per-pair cue maps


def patch_centres(gh: int, gw: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel rows and columns (gh, gw) of the patch centres of a token grid that covers the
    image uniformly (pixel centres at integers)."""
    rows = np.clip(np.round((np.arange(gh) + 0.5) * height / gh - 0.5), 0, height - 1)
    cols = np.clip(np.round((np.arange(gw) + 0.5) * width / gw - 0.5), 0, width - 1)
    r, c = np.meshgrid(rows.astype(np.int64), cols.astype(np.int64), indexing="ij")
    return r, c


def sample_tokens(tokens: np.ndarray, u: np.ndarray, v: np.ndarray, height: int,
                  width: int) -> np.ndarray:
    """L2-normalised bilinear samples (M, D) of a token grid (gh, gw, D) at pixel
    coordinates (u = column, v = row) of an image of `height` x `width`."""
    gh, gw = tokens.shape[:2]
    gx = np.clip((np.asarray(u, np.float64) + 0.5) * gw / width - 0.5, 0, gw - 1)
    gy = np.clip((np.asarray(v, np.float64) + 0.5) * gh / height - 0.5, 0, gh - 1)
    x0, y0 = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, gw - 1), np.minimum(y0 + 1, gh - 1)
    fx, fy = (gx - x0)[:, None], (gy - y0)[:, None]
    out = (tokens[y0, x0] * ((1 - fx) * (1 - fy)) + tokens[y0, x1] * (fx * (1 - fy))
           + tokens[y1, x0] * ((1 - fx) * fy) + tokens[y1, x1] * (fx * fy)).astype(np.float32)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


@dataclass
class PairMaps:
    """Cues of keyframe i's patches seen from one partner keyframe j (`pair_maps`)."""

    partner: int
    observed: np.ndarray  # (gh, gw) bool: j observed the patch's surface point
    geom: np.ndarray  # (gh, gw) bool: ... and saw through it (the surface is gone in j)
    feat: np.ndarray  # (gh, gw) float32: 1 - cos of the features where observed, else 0


def pair_maps(recon: Mapping[str, Any], i: int, j: int, tokens_i: np.ndarray,
              tokens_j: np.ndarray, params: Mapping[str, Any]) -> PairMaps:
    """Feature and depth cues of keyframe i's patch centres reprojected into keyframe j
    (module docstring, step 3). Token grids (gh, gw, D) as float32 or float16."""
    height, width = recon["depth"].shape[1:]
    gh, gw = tokens_i.shape[:2]
    rows, cols = patch_centres(gh, gw, height, width)
    z = recon["depth"][i][rows, cols].astype(np.float64)
    gr, gc = np.nonzero(recon["valid"][i][rows, cols] & (z > 0))
    observed, geom = np.zeros((gh, gw), bool), np.zeros((gh, gw), bool)
    feat = np.zeros((gh, gw), np.float32)
    if len(gr) == 0:
        return PairMaps(int(j), observed, geom, feat)
    K, T = recon["K"][i].astype(np.float64), recon["cam2world"][i].astype(np.float64)
    zz, r, c = z[gr, gc], rows[gr, gc], cols[gr, gc]
    cam = np.stack([(c - K[0, 2]) / K[0, 0] * zz, (r - K[1, 2]) / K[1, 1] * zz, zz], -1)
    u, v, zj = project_points(cam @ T[:3, :3].T + T[:3, 3], recon["K"][j].astype(np.float64),
                              se3_inverse(recon["cam2world"][j].astype(np.float64)))
    with np.errstate(invalid="ignore"):
        col, row = np.round(u), np.round(v)
        inside = (zj > 0) & (col >= 0) & (col < width) & (row >= 0) & (row < height)
    col = np.where(inside, col, 0).astype(np.int64)
    row = np.where(inside, row, 0).astype(np.int64)
    d = np.where(inside, recon["depth"][j][row, col], 0.0)
    seen = inside & recon["valid"][j][row, col] & (d > 0)
    tol = depth_tolerance(zj, params["covis_depth_rel"], params["covis_depth_abs"])
    obs = seen & (d > zj - tol)
    through = obs & (d - zj > params["geom_tol_abs"] + params["geom_tol_rel"] * zj)
    observed[gr, gc], geom[gr, gc] = obs, through
    if obs.any():
        fj = sample_tokens(tokens_j, u[obs], v[obs], height, width)
        fi = tokens_i[gr[obs], gc[obs]].astype(np.float32)
        fi /= np.maximum(np.linalg.norm(fi, axis=1, keepdims=True), 1e-12)
        feat[gr[obs], gc[obs]] = np.clip(1.0 - np.sum(fi * fj, axis=1), 0.0, 2.0)
    return PairMaps(int(j), observed, geom, feat)


@dataclass
class FrameScore:
    """Aggregated cues of one keyframe over its partners (`aggregate`)."""

    frame: int
    partners: list[int]  # partner keyframes (recon indices)
    score: np.ndarray  # (gh, gw) float32: geom + feat terms (0 with too few votes)
    feat: np.ndarray  # (gh, gw) float32: w_feat x mean feature dissimilarity
    geom: np.ndarray  # (gh, gw) float32: share of see-through votes
    votes: np.ndarray  # (gh, gw) int: partners that observed the patch
    voters: np.ndarray  # (P, gh, gw) bool: which partner observed it


def aggregate(frame: int, maps: Sequence[PairMaps], shape: tuple[int, int], w_feat: float,
              min_votes: int) -> FrameScore:
    """Score per patch = mean over the partners that observed it of (geom + w_feat feat);
    0 where fewer than `min_votes` (at least 1) partners observed it."""
    if not maps:
        zeros = np.zeros(shape, np.float32)
        return FrameScore(frame, [], zeros, zeros.copy(), zeros.copy(),
                          np.zeros(shape, np.int64), np.zeros((0, *shape), bool))
    voters = np.stack([m.observed for m in maps])
    votes = voters.sum(0)
    enough = votes >= max(min_votes, 1)
    n = np.maximum(votes, 1)
    feat = np.where(enough, w_feat * np.sum([m.feat for m in maps], 0) / n, 0.0)
    geom = np.where(enough, np.sum([m.geom for m in maps], 0) / n, 0.0)
    return FrameScore(frame, [m.partner for m in maps], (feat + geom).astype(np.float32),
                      feat.astype(np.float32), geom.astype(np.float32), votes, voters)


def upsample(grid: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear upsampling of a patch grid to the image (pixel-centre aligned)."""
    import cv2

    return cv2.resize(np.asarray(grid, np.float32), (width, height),
                      interpolation=cv2.INTER_LINEAR)


def max_entropy_threshold(values: np.ndarray, bins: int = 256) -> float | None:
    """Kapur's maximum-entropy threshold of the positive `values` (None if fewer than 2)."""
    values = np.asarray(values, np.float64).ravel()
    values = values[values > 0]
    if len(values) < 2 or values.max() <= 0:
        return None
    hist, edges = np.histogram(values, bins=bins, range=(0.0, float(values.max())))
    p = hist / hist.sum()
    cum = np.cumsum(p)
    best = None
    for t in range(bins - 1):
        pb, pf = cum[t], 1.0 - cum[t]
        if pb <= 0 or pf <= 0:
            continue
        b, f = p[:t + 1][p[:t + 1] > 0] / pb, p[t + 1:][p[t + 1:] > 0] / pf
        entropy = -float(np.sum(b * np.log(b)) + np.sum(f * np.log(f)))
        if best is None or entropy > best[0]:
            best = (entropy, float(edges[t + 1]))
    return None if best is None else best[1]


# ------------------------------------------------------------------ regions


@dataclass
class Region:
    """A connected high-score region of one keyframe."""

    rec: int
    frame: int
    mask: np.ndarray  # (H, W) bool
    box: tuple[int, int, int, int]  # x0, y0, x1, y1
    area: int
    score: float  # mean upsampled score over the region
    feat: float  # mean feature term
    geom: float  # mean depth term
    peaks: np.ndarray  # (k, 2) prompt points (x, y): highest-scoring patch centres inside
    partners: list[int]  # partner keyframes that observed some of its patches

    @property
    def feature_share(self) -> float:
        return self.feat / max(self.feat + self.geom, 1e-9)


def find_regions(
    rec: int, fs: FrameScore, height: int, width: int, threshold: float, min_px: int,
    max_px: int, n_points: int, max_regions: int,
) -> list[Region]:
    """Connected regions (8-neighbourhood) of the upsampled score >= `threshold` with
    `min_px` .. `max_px` pixels, highest mean score first (at most `max_regions`)."""
    score = upsample(fs.score, height, width)
    labels, n = ndimage.label(score >= threshold, structure=np.ones((3, 3), bool))
    if n == 0:
        return []
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    feat, geom = upsample(fs.feat, height, width), upsample(fs.geom, height, width)
    rows, cols = patch_centres(*fs.score.shape, height, width)
    out = []
    for k in np.flatnonzero((sizes >= min_px) & (sizes <= max_px)):
        if k == 0:
            continue
        mask = labels == k
        inside = mask[rows, cols] & (fs.score > 0)
        order = np.argsort(-fs.score[inside], kind="stable")[:n_points]
        peaks = np.stack([cols[inside][order], rows[inside][order]], 1).astype(np.float64)
        if len(peaks) == 0:  # thinner than the patch spacing: the region's best pixel
            y, x = np.unravel_index(np.argmax(np.where(mask, score, -np.inf)), mask.shape)
            peaks = np.array([[x, y]], np.float64)
        voted = [p for p, v in zip(fs.partners, fs.voters) if (v & inside).any()]
        out.append(Region(rec, fs.frame, mask, mask_box(mask), int(sizes[k]),
                          float(score[mask].mean()), float(feat[mask].mean()),
                          float(geom[mask].mean()), peaks, voted))
    out.sort(key=lambda r: -r.score)
    return out[:max_regions]


# ------------------------------------------------------------------ candidates


@dataclass
class Candidate:
    """A region segmented by SAM and lifted to 3D."""

    region: Region
    mask: np.ndarray  # (H, W) bool SAM mask
    sam_score: float
    iou: float  # IoU of the SAM mask with the region
    points: np.ndarray  # (M, 3) float32, 1 cm voxels
    colors: np.ndarray  # (M, 3) uint8
    static_fraction: float  # share of the points near the other recording's surfaces
    box: tuple[int, int, int, int]  # SAM prompt box


def segment_region(segmenter: Segmenter, recon: Mapping[str, Any], region: Region,
                   box_enlarge: float, min_iou: float
                   ) -> tuple[np.ndarray, float, float, tuple[int, int, int, int]] | None:
    """SAM mask of a region: prompt = the region box enlarged by `box_enlarge` plus its
    peak points; the best-scoring mask with IoU > `min_iou` with the region, as (mask,
    SAM score, IoU, prompt box), or None."""
    height, width = recon["depth"].shape[1:]
    box = enlarge_box(region.box, box_enlarge, width, height)
    masks, scores = segmenter.segment(str(region.frame), recon["images"][region.frame], box,
                                      region.peaks)
    best = None
    for mask, score in zip(masks, scores):
        iou = mask_iou(mask, region.mask)
        if iou > min_iou and mask.any() and (best is None or score > best[1]):
            best = (mask, float(score), iou, box)
    return best


def recording_cloud(recon: Mapping[str, Any], rec: int, voxel: float = STATIC_VOXEL
                    ) -> np.ndarray:
    """All valid points (M, 3) of one recording, voxel-downsampled per keyframe and then
    together (bounded memory)."""
    parts = [np.zeros((0, 3), np.float32)]
    for i in np.flatnonzero(recon["rec"] == rec):
        world = unproject_depth(recon["depth"][i], recon["K"][i], recon["cam2world"][i])
        parts.append(voxel_downsample(world[recon["valid"][i]], None, voxel)[0])
    return voxel_downsample(np.concatenate(parts), None, voxel)[0]


def near_fraction(points: np.ndarray, tree: cKDTree | None, dist: float) -> float:
    """Share of `points` within `dist` of the points of `tree` (0 without a tree)."""
    if tree is None or len(points) == 0 or tree.n == 0:
        return 0.0
    return float(np.isfinite(tree.query(points, distance_upper_bound=dist)[0]).mean())


def is_static(cand: Candidate, static_frac: float) -> bool:
    """The static-surface rule: mostly on the other recording's surfaces and not driven
    by the feature term (feature-driven candidates may lie flat on a surface)."""
    return cand.static_fraction > static_frac and cand.region.feature_share <= FEATURE_DRIVEN


def cluster_candidates(cands: Sequence[Candidate], dist: float, frac: float
                       ) -> list[list[int]]:
    """Union-find groups of candidates (one recording) whose point sets overlap: more than
    `frac` of the points of one within `dist` of the other (`point_overlap`)."""
    parent = list(range(len(cands)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lo = [c.points.min(0) for c in cands]
    hi = [c.points.max(0) for c in cands]
    for a in range(len(cands)):
        for b in range(a + 1, len(cands)):
            if root(a) == root(b) or np.any(lo[a] - dist > hi[b]) or np.any(lo[b] - dist > hi[a]):
                continue
            if point_overlap(cands[a].points, cands[b].points, dist) > frac:
                parent[root(b)] = root(a)
    groups: dict[int, list[int]] = {}
    for i in range(len(cands)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def consensus_points(
    members: Sequence[Candidate], recon: Mapping[str, Any], voxel: float, rel: float,
    abs_: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Points and colours of a cluster: the union of its candidates' points (`voxel`
    cells, outliers removed), minus the points that fewer than `CONSENSUS_SHARE` of the
    member keyframes put inside their mask among those that see them (a point is seen in
    a keyframe if it projects onto a valid pixel with a depth within max(abs_, rel z), or
    if that keyframe's mask holds it). Without `CONSENSUS_MIN_FRAMES` keyframes no vote is
    taken. Also returns the number of points dropped."""
    points, colors = clean_points(np.concatenate([c.points for c in members]),
                                  np.concatenate([c.colors for c in members]), voxel)
    frames = sorted({c.region.frame for c in members})
    if len(frames) < CONSENSUS_MIN_FRAMES or len(points) == 0:
        return points, colors, 0
    height, width = recon["depth"].shape[1:]
    inside_masks = np.zeros(len(points), np.int64)
    seen = np.zeros(len(points), np.int64)
    for frame in frames:
        held = np.zeros(len(points), bool)
        for c in members:
            if c.region.frame == frame and len(c.points):
                dist = cKDTree(c.points).query(points, distance_upper_bound=1.5 * voxel)[0]
                held |= np.isfinite(dist)
        col, row, z, inside = project_to_pixels(points, recon["K"][frame],
                                                recon["cam2world"][frame], width, height)
        idx = np.flatnonzero(inside)
        d = recon["depth"][frame][row[idx], col[idx]]
        visible = np.zeros(len(points), bool)
        visible[idx] = recon["valid"][frame][row[idx], col[idx]] & (
            np.abs(d - z[idx]) < depth_tolerance(z[idx], rel, abs_))
        inside_masks += held
        seen += held | visible
    keep = inside_masks >= CONSENSUS_SHARE * np.maximum(seen, 1)
    return points[keep], colors[keep], int((~keep).sum())


def cluster_views(members: Sequence[Candidate]) -> list[View]:
    """One view per keyframe of a cluster (masks of the same keyframe united), best
    SAM score first."""
    views: dict[int, View] = {}
    for c in members:
        frame = c.region.frame
        if frame in views:
            old = views[frame]
            box = (min(old.box[0], c.box[0]), min(old.box[1], c.box[1]),
                   max(old.box[2], c.box[2]), max(old.box[3], c.box[3]))
            views[frame] = View(frame, box, old.mask | c.mask, max(old.score, c.sam_score))
        else:
            views[frame] = View(frame, c.box, c.mask, c.sam_score)
    return sorted(views.values(), key=lambda v: -v.score)


def free_evidence(points: np.ndarray, grid: Mapping[str, Any] | None, other: int
                  ) -> tuple[int, float]:
    """Free-space evidence of the other recording over the M3 voxels holding `points`:
    (sum, mean) of its free counts (0 without a grid)."""
    if grid is None or len(points) == 0:
        return 0, 0.0
    free = grid[f"free_count_{RECORDINGS[other]}"]
    ijk = np.unique(np.floor((points - grid["origin"]) / grid["voxel_size"]).astype(np.int64),
                    axis=0)
    ijk = ijk[np.all((ijk >= 0) & (ijk < np.asarray(free.shape)), axis=1)]
    if len(ijk) == 0:
        return 0, 0.0
    counts = free[tuple(ijk.T)].astype(np.int64)
    return int(counts.sum()), round(float(counts.mean()), 2)


def cluster_evidence(members: Sequence[Candidate], points: np.ndarray,
                     recon: Mapping[str, Any], rec: int, grid: Mapping[str, Any] | None
                     ) -> dict[str, Any]:
    """Evidence of a cluster: support and cue terms of its regions, the other
    recording's views of the place (not occluded, >= MIN_VISIBLE_FRAC of the points) and
    its free-space counts there, distance to the nearest camera of the other recording."""
    other = 1 - rec
    frames = np.flatnonzero(recon["rec"] == other)
    views = score_views(points, frames, recon["cam2world"], recon["K"], recon["depth"],
                        recon["valid"], PLACE_DEPTH_TOL, on_surface=False)
    free_rays, free_mean = free_evidence(points, grid, other)
    centres = recon["cam2world"][frames, :3, 3].astype(np.float64)
    centroid = points.mean(0)
    distance = float(np.linalg.norm(centres - centroid, axis=1).min()) if len(centres) else None
    regions = [c.region for c in members]
    return {
        "frames_supporting": len({r.frame for r in regions}),
        "mean_score": round(float(np.mean([r.score for r in regions])), 4),
        "feat_term": round(float(np.mean([r.feat for r in regions])), 4),
        "geom_term": round(float(np.mean([r.geom for r in regions])), 4),
        "partners_visible": len({p for r in regions for p in r.partners}),
        "observing_views": sum(v.visible >= MIN_VISIBLE_FRAC for v in views),
        "free_rays": free_rays,
        "free_frames_mean": free_mean,
        "min_camera_distance": None if distance is None else round(distance, 3),
        "static_fraction": round(float(np.mean([c.static_fraction for c in members])), 4),
    }


# ------------------------------------------------------------------ instances


APPEARANCE_KEYS = ("frames_supporting", "mean_score", "feat_term", "geom_term",
                   "partners_visible")


def instance_evidence(cluster: Mapping[str, Any]) -> dict[str, Any]:
    """M4-style instance evidence of a cluster (`roomdiff.instances.instance_record`) plus
    the appearance terms under `appearance`."""
    ev = cluster["evidence"]
    obb = cluster["obb"]
    return {
        "voxels": 0,
        "volume_m3": round(float(np.prod(obb["extents"])), 6),
        "free_rays": ev["free_rays"],
        "free_frames_mean": ev["free_frames_mean"],
        "observing_views": ev["observing_views"],
        "occ_points": cluster["n_points"],
        "min_camera_distance": ev["min_camera_distance"],
        "appearance": {"cluster": cluster["id"], **{k: ev[k] for k in APPEARANCE_KEYS}},
    }


def load_clusters(out_dir: str | Path) -> tuple[list[dict[str, Any]], dict[int, np.ndarray],
                                                dict[int, np.ndarray],
                                                dict[int, list[tuple[int, dict[str, Any]]]]]:
    """Clusters of `<out>/appearance/`: records, points / colours and (frame, RLE) masks
    per cluster id."""
    folder = OutputPaths(out_dir).appearance
    records = json.loads((folder / "clusters.json").read_text())["clusters"]
    with np.load(folder / "clusters_points.npz") as f:
        points, colors, owner = f["points"], f["colors"], f["cluster"]
    masks: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    with np.load(folder / "clusters_masks.npz") as f:
        size = [int(s) for s in f["size"]]
        for cid, frame, counts in zip(f["cluster"], f["frame"], f["rle"]):
            masks.setdefault(int(cid), []).append((int(frame), {"size": size,
                                                               "counts": str(counts)}))
    ids = [r["id"] for r in records]
    return (records, {i: points[owner == i] for i in ids}, {i: colors[owner == i] for i in ids},
            masks)


def load_appearance_instances(
    out_dir: str | Path, existing: Sequence[Sequence[Instance]], covered_dist: float,
    covered_frac: float,
) -> tuple[list[list[Instance]], dict[str, Any]]:
    """Instances (per recording) from the appearance clusters that the `existing`
    instances of their recording do not cover: more than `covered_frac` of the cluster's
    points within `covered_dist` of an existing instance's points means covered. The new
    instances (kind disappeared / appeared, `source` "appearance", ids still 0) keep the
    cluster's views; features and per-keyframe masks are computed by the caller.
    Returns them and a meta record."""
    records, points, colors, masks = load_clusters(out_dir)
    per_rec: list[list[Instance]] = [[] for _ in RECORDINGS]
    covered, added = [], []
    for record in records:
        rec = RECORDINGS.index(record["recording"])
        pts = points[record["id"]]
        overlaps = [(near_fraction(pts, cKDTree(i.points), covered_dist), i.id)
                    for i in existing[rec] if len(i.points)]
        best = max(overlaps, default=(0.0, None))
        if best[0] > covered_frac:
            covered.append({"cluster": record["id"], "recording": record["recording"],
                            "instance": best[1], "overlap": round(best[0], 3)})
            continue
        boxes = {v["frame"]: v for v in record["views"]}
        views = [View(frame, tuple(boxes[frame]["box"]), decode_rle(rle),
                      boxes[frame]["score"]) for frame, rle in masks.get(record["id"], [])]
        views.sort(key=lambda v: -v.score)
        inst = Instance(rec, KINDS[rec], [], pts, pts.astype(np.float32),
                        colors[record["id"]], views, source="appearance",
                        evidence=instance_evidence(record))
        per_rec[rec].append(inst)
        added.append({"cluster": record["id"], "recording": record["recording"],
                      "overlap": round(best[0], 3)})
    for instances in per_rec:  # largest first, like the geometric instances
        instances.sort(key=lambda i: -len(i.points))
    return per_rec, {"clusters": len(records), "covered": covered, "added": added}


# ------------------------------------------------------------------ stage


def _peak_memory() -> dict[str, Any]:
    torch = sys.modules.get("torch")
    vram = None
    if torch is not None and torch.cuda.is_initialized():
        vram = {"allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    return {"peak_vram_gb": vram,
            "peak_ram_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}


def _reset_vram_peak() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()


def keyframe_tokens(featurizer: Featurizer, recon: Mapping[str, Any], frames: Sequence[int]
                    ) -> np.ndarray:
    """Patch tokens (n, gh, gw, D) float16 of the keyframes `frames`."""
    out: np.ndarray | None = None
    for k, i in enumerate(frames):
        tokens = featurizer.patch_tokens(f"appearance:{i}", recon["images"][i])
        if out is None:
            out = np.zeros((len(frames), *tokens.shape), np.float16)
        out[k] = tokens
    return out if out is not None else np.zeros((0, 1, 1, featurizer.dim), np.float16)


def cached_tokens(folder: Path, featurizer: Featurizer, recon: Mapping[str, Any],
                  frames: Sequence[int], name: str) -> tuple[np.ndarray, bool]:
    """Tokens of one recording from `feats_<name>.npy` if it was written by the same model
    for the same keyframes (`feats_<name>.json`), else computed and cached. Returns
    (tokens, cache hit)."""
    model = getattr(featurizer, "model_id", type(featurizer).__name__)
    key = {"model": model, "files": [str(recon["files"][i]) for i in frames],
           "image_size": list(recon["images"].shape[1:3])}
    npy, info = folder / f"feats_{name}.npy", folder / f"feats_{name}.json"
    if npy.is_file() and info.is_file() and json.loads(info.read_text()) == key:
        return np.load(npy), True
    tokens = keyframe_tokens(featurizer, recon, frames)
    np.save(npy, tokens)
    info.write_text(json.dumps(key))
    return tokens, False


def run_appearance(
    out_dir: str | Path, cfg: Mapping[str, Any], segmenter: Segmenter | None = None,
    featurizer: Featurizer | None = None,
) -> dict[str, Any]:
    """Appearance change clusters of `<out>/recon/` (module docstring); writes
    `<out>/appearance/`. Real models (from the `instances` section) are loaded unless
    both are given. Returns meta.json (only `{"enabled": false}` when disabled)."""
    params = dict(cfg["appearance"])
    if not params["enabled"]:
        print("[appearance] disabled (appearance.enabled = false)")
        return {"enabled": False}
    inst_params = dict(cfg["instances"])
    paths = OutputPaths(out_dir)
    folder = paths.appearance
    folder.mkdir(parents=True, exist_ok=True)
    times: dict[str, float] = {}
    start = time.perf_counter()
    _reset_vram_peak()
    recon = load_recon(paths.recon / "recon.npz")
    grid_file = paths.visibility / "grid.npz"
    grid = None
    if grid_file.is_file():
        from roomdiff.visibility import load_grid

        grid = load_grid(grid_file)
    height, width = recon["depth"].shape[1:]
    frames = [np.flatnonzero(recon["rec"] == r) for r in range(len(RECORDINGS))]
    times["load"] = time.perf_counter() - start

    t0 = time.perf_counter()
    covis = covisibility(recon, frames[0], frames[1], params["covis_stride"],
                         params["covis_depth_rel"], params["covis_depth_abs"])
    partners = [select_partners(covis, params["pairs_per_frame"], params["min_covisibility"]),
                select_partners(covis.T, params["pairs_per_frame"], params["min_covisibility"])]
    times["covisibility"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if segmenter is None or featurizer is None:
        from roomdiff.models import load_models

        segmenter, featurizer = load_models(inst_params)
        _reset_vram_peak()
    times["load_models"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tokens, cache_hits = [], []
    for r, name in enumerate(RECORDINGS):
        t, hit = cached_tokens(folder, featurizer, recon, frames[r], name)
        tokens.append(t)
        cache_hits.append(hit)
    times["features"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores: list[list[FrameScore]] = [[], []]
    for r in range(len(RECORDINGS)):
        other = 1 - r
        for k, i in enumerate(frames[r]):
            maps = [pair_maps(recon, int(i), int(frames[other][m]),
                              tokens[r][k].astype(np.float32),
                              tokens[other][m].astype(np.float32), params)
                    for m in partners[r][k]]
            scores[r].append(aggregate(int(i), maps, tokens[r].shape[1:3], params["w_feat"],
                                       params["min_votes"]))
    all_scores = np.concatenate([fs.score.ravel() for s in scores for fs in s] or [np.zeros(0)])
    auto = max_entropy_threshold(all_scores)
    threshold = auto if params["threshold_mode"] == "max_entropy" and auto is not None \
        else params["score_threshold"]
    times["score_maps"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    min_px = max(1, round(params["min_region_frac"] * height * width))
    max_px = round(params["max_region_frac"] * height * width)
    regions = [[reg for fs in scores[r] for reg in find_regions(
        r, fs, height, width, threshold, min_px, max_px, params["n_prompt_points"],
        params["max_regions_per_frame"])] for r in range(len(RECORDINGS))]
    times["regions"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    trees = [cKDTree(recording_cloud(recon, r)) for r in range(len(RECORDINGS))]
    region_log: list[dict[str, Any]] = []
    candidates: list[list[Candidate]] = [[], []]
    for r in range(len(RECORDINGS)):
        for reg in regions[r]:
            entry = {"recording": RECORDINGS[r], "frame": reg.frame,
                     "file": str(recon["files"][reg.frame]), "box": list(reg.box),
                     "area": reg.area, "score": round(reg.score, 4), "feat": round(reg.feat, 4),
                     "geom": round(reg.geom, 4), "partners": reg.partners}
            region_log.append(entry)
            picked = segment_region(segmenter, recon, reg, params["box_enlarge"],
                                    params["min_mask_iou"])
            if picked is None:
                entry["outcome"] = "no SAM mask overlapping the region"
                continue
            mask, sam_score, iou, box = picked
            entry.update(sam_score=round(sam_score, 4), mask_iou=round(iou, 4))
            pts, cols = lift_mask(recon, reg.frame, mask)
            pts, cols = clean_points(pts, cols, inst_params["lift_voxel"])
            if len(pts) < params["min_points"]:
                entry["outcome"] = f"{len(pts)} points < min_points"
                continue
            cand = Candidate(reg, mask, sam_score, iou, pts, cols,
                             near_fraction(pts, trees[1 - r], params["static_dist"]), box)
            entry["static_fraction"] = round(cand.static_fraction, 4)
            if is_static(cand, params["static_frac"]):
                entry["outcome"] = "static surface"
                continue
            entry["outcome"] = "candidate"
            entry["candidate"] = len(candidates[r])
            candidates[r].append(cand)
    del trees
    times["segment_and_lift"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    clusters: list[dict[str, Any]] = []
    cluster_points, cluster_colors, cluster_owner, mask_rows = [], [], [], []
    dropped_clusters = []
    for r in range(len(RECORDINGS)):
        groups = cluster_candidates(candidates[r], params["cluster_dist"], params["cluster_frac"])
        for group in groups:
            members = [candidates[r][k] for k in group]
            support = len({c.region.frame for c in members})
            if support < params["min_frames"]:
                dropped_clusters.append({"recording": RECORDINGS[r], "candidates": group,
                                         "frames": support})
                continue
            pts, cols, dropped = consensus_points(members, recon, inst_params["lift_voxel"],
                                                  params["covis_depth_rel"],
                                                  params["covis_depth_abs"])
            if len(pts) < params["min_points"]:
                dropped_clusters.append({"recording": RECORDINGS[r], "candidates": group,
                                         "frames": support, "points": len(pts)})
                continue
            cid = len(clusters) + 1
            views = cluster_views(members)
            region_of = {}
            for c in members:
                best = region_of.get(c.region.frame)
                if best is None or c.region.score > best.score:
                    region_of[c.region.frame] = c.region
            lo, hi = pts.min(0), pts.max(0)
            clusters.append({
                "id": cid, "recording": RECORDINGS[r], "kind": KINDS[r],
                "candidates": group, "n_points": len(pts), "consensus_dropped": dropped,
                "centroid": pts.mean(0).round(4).tolist(),
                "aabb": [lo.round(4).tolist(), hi.round(4).tolist()],
                "extents": (hi - lo).round(4).tolist(), "obb": floor_obb(pts),
                "views": [{"frame": v.frame, "file": str(recon["files"][v.frame]),
                           "box": [int(b) for b in v.box], "mask_box": list(mask_box(v.mask)),
                           "score": round(v.score, 4),
                           "region_score": round(region_of[v.frame].score, 4),
                           "feature_share": round(region_of[v.frame].feature_share, 4)}
                          for v in views],
                "evidence": cluster_evidence(members, pts, recon, r, grid),
            })
            cluster_points.append(pts)
            cluster_colors.append(cols)
            cluster_owner.append(np.full(len(pts), cid, np.int32))
            mask_rows += [(cid, v.frame, encode_rle(v.mask)["counts"]) for v in views]
    times["cluster"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for r, name in enumerate(RECORDINGS):
        fs = scores[r]
        shape = tokens[r].shape[1:3]
        width_p = params["pairs_per_frame"]
        partner_ids = np.full((len(fs), width_p), -1, np.int32)
        for k, f in enumerate(fs):
            partner_ids[k, :len(f.partners)] = f.partners
        np.savez_compressed(
            folder / f"scores_{name}.npz",
            frames=np.array([f.frame for f in fs], np.int32), partners=partner_ids,
            score=np.stack([f.score for f in fs]).astype(np.float16) if fs
            else np.zeros((0, *shape), np.float16),
            feat=np.stack([f.feat for f in fs]).astype(np.float16) if fs
            else np.zeros((0, *shape), np.float16),
            geom=np.stack([f.geom for f in fs]).astype(np.float16) if fs
            else np.zeros((0, *shape), np.float16),
            votes=np.stack([f.votes for f in fs]).astype(np.uint8) if fs
            else np.zeros((0, *shape), np.uint8),
            threshold=np.float64(threshold))
    pairs_doc = {
        "frames": {name: [int(i) for i in frames[r]] for r, name in enumerate(RECORDINGS)},
        "covisibility": np.round(covis, 3).tolist(),
        "partners": {name: {str(int(frames[r][k])): [int(frames[1 - r][m]) for m in ps]
                            for k, ps in enumerate(partners[r])}
                     for r, name in enumerate(RECORDINGS)},
    }
    (folder / "pairs.json").write_text(json.dumps(pairs_doc))
    (folder / "regions.json").write_text(json.dumps({"regions": region_log}, indent=1))
    (folder / "clusters.json").write_text(json.dumps(
        {"image_size": [int(width), int(height)], "threshold": threshold,
         "clusters": clusters}, indent=1))
    empty = [np.zeros((0, 3))]
    np.savez_compressed(
        folder / "clusters_points.npz",
        points=np.concatenate(cluster_points or empty).astype(np.float32),
        colors=np.concatenate(cluster_colors or empty).astype(np.uint8),
        cluster=np.concatenate(cluster_owner or [np.zeros(0)]).astype(np.int32))
    np.savez_compressed(
        folder / "clusters_masks.npz",
        cluster=np.array([m[0] for m in mask_rows], np.int32),
        frame=np.array([m[1] for m in mask_rows], np.int32),
        rle=np.array([m[2] for m in mask_rows], dtype=np.str_),
        size=np.array([height, width], np.int64))
    times["write"] = time.perf_counter() - t0
    times["total"] = time.perf_counter() - start

    outcomes: dict[str, int] = {}
    for entry in region_log:
        reason = entry["outcome"] if not entry["outcome"].endswith("< min_points") \
            else "too few points"
        outcomes[reason] = outcomes.get(reason, 0) + 1
    paired = [sum(1 for p in partners[r] if p) for r in range(len(RECORDINGS))]
    meta = {
        "enabled": True,
        "params": params,
        "models": {"sam": inst_params["sam_model"],
                   "dino": getattr(featurizer, "model_id", type(featurizer).__name__)},
        "token_grid": list(tokens[0].shape[1:3]) if len(tokens[0]) else None,
        "features_cached": dict(zip(RECORDINGS, cache_hits)),
        "keyframes_with_partners": dict(zip(RECORDINGS, paired)),
        "covisibility_max": round(float(covis.max()), 3) if covis.size else None,
        "threshold": {"used": round(float(threshold), 4), "fixed": params["score_threshold"],
                      "max_entropy": None if auto is None else round(auto, 4),
                      "mode": params["threshold_mode"]},
        "score_stats": {
            "patches_scored": int((all_scores > 0).sum()),
            "above_threshold": int((all_scores >= threshold).sum()),
            "p50": round(float(np.percentile(all_scores[all_scores > 0], 50)), 4)
            if (all_scores > 0).any() else None,
            "p95": round(float(np.percentile(all_scores[all_scores > 0], 95)), 4)
            if (all_scores > 0).any() else None},
        "regions": {name: len(regions[r]) for r, name in enumerate(RECORDINGS)},
        "region_outcomes": outcomes,
        "candidates": {name: len(candidates[r]) for r, name in enumerate(RECORDINGS)},
        "clusters": {name: sum(c["recording"] == name for c in clusters)
                     for name in RECORDINGS},
        "clusters_dropped": dropped_clusters,
        "runtime_s": {k: round(v, 2) for k, v in times.items()},
        **_peak_memory(),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[appearance] threshold {threshold:.3f} (max-entropy "
          f"{'n/a' if auto is None else f'{auto:.3f}'}): regions "
          f"{meta['regions']['before']} / {meta['regions']['after']} -> candidates "
          f"{meta['candidates']['before']} / {meta['candidates']['after']} -> clusters "
          f"{meta['clusters']['before']} / {meta['clusters']['after']}, {times['total']:.1f} s"
          f" -> {folder}")
    return meta
