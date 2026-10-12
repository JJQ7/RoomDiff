"""Project world points and boxes into keyframes; rank the keyframes that see an object.

Shared by M3 (debug crops, observing views) and M4 (SAM prompting views, crops). numpy
only, same conventions as `roomdiff.geometry` (OpenCV cameras, pixel centres at integers).

View ranking (`score_views`, `best_views`) depends only on the object's own points as
seen by each camera, never on the world axes, so turning the world frame (e.g. snapping
it to the walls) does not change which keyframes are picked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from roomdiff.geometry import project_points, se3_inverse

MAX_SCORED_POINTS = 2000  # points projected per view (fixed random subset of the object)
MIN_VISIBLE_FRAC = 0.15  # a view must see at least this fraction of the points
BORDER_PX = 3  # projected points this close to the image border: the object is cut off
CUT_FACTOR = 0.8  # score multiplier of a view in which the object is cut by the border
FRONTAL_WEIGHT = 0.2  # largest score penalty for looking along the object's long axis


def project_to_pixels(
    points: np.ndarray, K: np.ndarray, cam2world: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearest pixel `(col, row)` (int, -1 outside), camera z-depth and in-image mask of
    world points (M, 3). Points behind the camera or outside the image are not inside."""
    u, v, z = project_points(np.asarray(points, np.float64), np.asarray(K, np.float64),
                             se3_inverse(np.asarray(cam2world, np.float64)))
    with np.errstate(invalid="ignore"):
        col, row = np.round(u), np.round(v)
        inside = (z > 0) & (col >= 0) & (col < width) & (row >= 0) & (row < height)
    col = np.where(inside, col, -1).astype(np.int64)
    row = np.where(inside, row, -1).astype(np.int64)
    return col, row, z, inside


def aabb_corners(lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
    """The 8 corners (8, 3) of an axis-aligned box."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    pick = np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)], bool)
    return np.where(pick, hi, lo)


def project_aabb(
    lo: Sequence[float], hi: Sequence[float], K: np.ndarray, cam2world: np.ndarray,
    width: int, height: int,
) -> tuple[int, int, int, int] | None:
    """2D bounding box `(x0, y0, x1, y1)` of the projected corners of a 3D box, clipped to
    the image. Corners behind the camera are ignored; None if nothing is visible."""
    u, v, z = project_points(aabb_corners(lo, hi), np.asarray(K, np.float64),
                             se3_inverse(np.asarray(cam2world, np.float64)))
    front = z > 1e-3
    if not front.any():
        return None
    x0, x1 = np.clip([u[front].min(), u[front].max()], 0, width - 1)
    y0, y1 = np.clip([v[front].min(), v[front].max()], 0, height - 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))


def subsample(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """At most `n` of `points`: a fixed random subset of the indices (not of the
    coordinates, so the same points are picked in any world frame)."""
    if len(points) <= n:
        return points
    return points[np.random.default_rng(seed).choice(len(points), n, replace=False)]


def horizontal_axis(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Dominant horizontal direction (unit (x, z)) of a point set and its elongation in
    [0, 1] (1 - minor / major variance of the x-z spread; 0 for a round footprint).
    Assumes a vertical y axis (levelled frame; roughly true for the camera frame too)."""
    xz = np.asarray(points, np.float64)[:, [0, 2]]
    if len(xz) < 2:
        return np.array([1.0, 0.0]), 0.0
    xz = xz - xz.mean(0)
    values, vectors = np.linalg.eigh(xz.T @ xz / len(xz))
    if values[1] <= 1e-12:
        return np.array([1.0, 0.0]), 0.0
    return vectors[:, 1], float(1.0 - max(values[0], 0.0) / values[1])


@dataclass(frozen=True)
class ViewScore:
    """How well one keyframe shows an object (`score_views`)."""

    frame: int
    score: float  # visible box area (px) x visible x border factor x frontal factor
    visible: float  # fraction of the points visible in the view
    in_image: float  # fraction of the points projecting inside the image
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 of all points projecting inside the image
    cut: bool  # the projected object touches the image border

    @property
    def occluded(self) -> bool:
        return self.visible < MIN_VISIBLE_FRAC


def score_views(
    points: np.ndarray,
    frames: Sequence[int],
    cam2world: np.ndarray,
    K: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray | None = None,
    depth_tol: float = 0.05,
    on_surface: bool = True,
) -> list[ViewScore]:
    """Rank `frames` by how well they show the object made of `points` (M, 3), best first.

    A point is visible in a view if it projects inside the image onto a valid pixel and,
    with `on_surface` (the object exists in that recording), the observed depth is
    within `depth_tol` of the point's depth (depth-consistent); without it (the other
    recording looking at the place) no surface lies more than `depth_tol` in front of it.
    Score = area of the 2D box of the visible points x fraction of the points visible x
    `CUT_FACTOR` if the projected object touches the image border x a frontal factor
    `1 - FRONTAL_WEIGHT * e * cos^2` (e = elongation of the footprint, angle between the
    horizontal viewing ray and the object's long horizontal axis). Everything is measured
    in the images, so the ranking does not depend on the world frame. Frames in which no
    point projects inside the image are left out. At most `MAX_SCORED_POINTS` points are
    used (a fixed subset).
    """
    pts = subsample(np.asarray(points, np.float64).reshape(-1, 3), MAX_SCORED_POINTS)
    if len(pts) == 0:
        return []
    height, width = depth.shape[1:]
    centroid = pts.mean(0)
    axis, elongation = horizontal_axis(pts)
    out = []
    for i in frames:
        col, row, z, inside = project_to_pixels(pts, K[i], cam2world[i], width, height)
        if not inside.any():
            continue
        observed = np.zeros(len(pts))
        observed[inside] = depth[i][row[inside], col[inside]]
        ok = inside & (observed > 0)
        if valid is not None:
            ok[inside] &= valid[i][row[inside], col[inside]]
        if on_surface:
            visible = ok & (np.abs(observed - z) <= depth_tol)
        else:
            visible = ok & (observed >= z - depth_tol)
        c, r = col[inside], row[inside]
        box = (int(c.min()), int(r.min()), int(c.max()), int(r.max()))
        cut = bool(min(box[0], box[1]) <= BORDER_PX or box[2] >= width - 1 - BORDER_PX
                   or box[3] >= height - 1 - BORDER_PX)
        frac = float(visible.mean())
        area = 0.0
        if visible.any():
            cv, rv = col[visible], row[visible]
            area = float((cv.max() - cv.min() + 1) * (rv.max() - rv.min() + 1))
        ray = (centroid - cam2world[i][:3, 3])[[0, 2]]
        cos = abs(float(ray @ axis)) / max(float(np.linalg.norm(ray)), 1e-9)
        frontal = 1.0 - FRONTAL_WEIGHT * elongation * cos**2
        score = area * frac * (CUT_FACTOR if cut else 1.0) * frontal
        out.append(ViewScore(int(i), score, frac, float(inside.mean()), box, cut))
    out.sort(key=lambda s: (-s.score, -s.in_image, s.frame))
    return out


def best_views(
    points: np.ndarray,
    frames: Sequence[int],
    cam2world: np.ndarray,
    K: np.ndarray,
    depth: np.ndarray,
    k: int = 2,
    depth_tol: float = 0.05,
    valid: np.ndarray | None = None,
    on_surface: bool = True,
) -> list[ViewScore]:
    """The `k` best of `frames` for the object `points` (`score_views`), best first;
    views that see none of it come last (by their in-image share)."""
    return score_views(points, frames, cam2world, K, depth, valid, depth_tol, on_surface)[:k]
