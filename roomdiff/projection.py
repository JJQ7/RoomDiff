"""Project world points and boxes into keyframes; pick the keyframes that see a point best.

Shared by M3 (debug crops, observing views) and M4 (best views per object). numpy only,
same conventions as `roomdiff.geometry` (OpenCV cameras, pixel centres at integers).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from roomdiff.geometry import project_points, se3_inverse


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


def view_angle_deg(points: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    """Angle (degrees) between the optical axis of `cam2world` and the rays to `points`."""
    rays = np.asarray(points, np.float64) - cam2world[:3, 3]
    cos = rays @ cam2world[:3, 2] / np.maximum(np.linalg.norm(rays, axis=-1), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


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


def best_views(
    point: Sequence[float],
    frames: Sequence[int],
    cam2world: np.ndarray,
    K: np.ndarray,
    depth: np.ndarray,
    k: int = 2,
    occlusion_tol: float = 0.10,
) -> list[tuple[int, bool]]:
    """Up to `k` of `frames` that see the world `point` best: inside the image and with
    the smallest angle to the optical axis. Frames whose depth map puts a surface more
    than `occlusion_tol` in front of the point are used only if too few others exist.

    Returns `(frame, occluded)` pairs, best first.
    """
    height, width = depth.shape[1:]
    point = np.asarray(point, np.float64)[None]
    scored = []
    for i in frames:
        col, row, z, inside = project_to_pixels(point, K[i], cam2world[i], width, height)
        if not inside[0]:
            continue
        occluded = bool(z[0] > depth[i, row[0], col[0]] + occlusion_tol)
        scored.append((occluded, float(view_angle_deg(point, cam2world[i])[0]), int(i)))
    scored.sort()
    return [(i, occluded) for occluded, _, i in scored[:k]]
