"""Small numpy geometry helpers (CPU only).

Conventions: OpenCV cameras (x right, y down, z forward); `cam2world` / `world2cam` are
4x4 rigid transforms; pixel (u, v) = (column, row) with pixel centres at integer
coordinates, the convention of `pi3.utils.geometry.recover_intrinsic_from_rays_d`, so an
intrinsic matrix recovered from Pi3X rays can be used here directly.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

MIN_YAW_SPREAD = 0.1  # min singular value ratio of the camera x axes to trust 2 directions


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Inverse of one or a batch (..., 4, 4) of rigid transforms."""
    R = T[..., :3, :3]
    t = T[..., :3, 3:]
    Rt = np.swapaxes(R, -1, -2)
    inv = np.zeros_like(T)
    inv[..., :3, :3] = Rt
    inv[..., :3, 3:] = -Rt @ t
    inv[..., 3, 3] = 1
    return inv


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform (rigid or similarity) to points (..., 3)."""
    return points @ T[:3, :3].T + T[:3, 3]


def unproject_depth(depth: np.ndarray, K: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    """World points (H, W, 3) of a z-depth map (H, W) seen by camera `K`, `cam2world`."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    cam = np.stack([x * depth, y * depth, depth], axis=-1)
    return transform_points(cam2world.astype(np.float32), cam).astype(np.float32)


def project_points(
    points_world: np.ndarray, K: np.ndarray, world2cam: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points (..., 3) to pixel coordinates; returns `(u, v, z)`.

    `z` is the depth in the camera frame; points with z <= 0 get u = v = nan.
    """
    cam = transform_points(world2cam, points_world)
    z = cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_z = np.where(z > 0, z, np.nan)
        u = K[0, 0] * cam[..., 0] / safe_z + K[0, 2]
        v = K[1, 1] * cam[..., 1] / safe_z + K[1, 2]
    return u, v, z


def umeyama(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form least-squares similarity `dst ~ s * R @ src + t` (Umeyama 1991).

    `src`, `dst`: (N, 3) corresponding points, N >= 3 and not collinear.
    Returns `(s, R, t)`; `s` is 1.0 when `with_scale` is False.
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:  # avoid a reflection
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / (xs**2).sum(1).mean()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def sim3_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4x4 matrix of the similarity `x -> s * R @ x + t`."""
    M = np.eye(4)
    M[:3, :3] = s * np.asarray(R)
    M[:3, 3] = t
    return M


def sim3_apply_to_poses(M: np.ndarray, cam2world: np.ndarray) -> tuple[np.ndarray, float]:
    """Move cameras (N, 4, 4) by the similarity `M`; returns `(new cam2world, scale)`.

    Rotations stay orthonormal; depth maps of the moved cameras must be multiplied by the
    returned scale.
    """
    s = float(np.cbrt(np.linalg.det(M[:3, :3])))
    R = M[:3, :3] / s
    out = cam2world.astype(np.float64).copy()
    out[:, :3, :3] = R @ out[:, :3, :3]
    out[:, :3, 3] = out[:, :3, 3] @ (s * R).T + M[:3, 3]
    return out.astype(cam2world.dtype), s


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray | None, voxel: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """Average the points (and colours) falling into each cubic voxel of size `voxel`.

    Returns float32 points (M, 3) and, if given, colours (M, C) in their input dtype.
    """
    if len(points) == 0:
        return points.astype(np.float32), colors
    idx = np.floor(points / voxel).astype(np.int64)
    idx -= idx.min(0)
    dims = idx.max(0) + 1
    if np.prod(dims.astype(np.float64)) < 2**62:  # one int64 key per voxel: fast unique
        key = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
        _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    else:  # absurd extents (far outliers): slower row-wise unique
        _, inverse, counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()

    def mean(values: np.ndarray) -> np.ndarray:
        sums = np.stack(
            [np.bincount(inverse, weights=values[:, c]) for c in range(values.shape[1])], 1
        )
        return sums / counts[:, None]

    out_points = mean(points.astype(np.float64)).astype(np.float32)
    if colors is None:
        return out_points, None
    out_colors = mean(colors.astype(np.float64))
    if np.issubdtype(colors.dtype, np.integer):
        out_colors = np.round(out_colors)
    return out_points, out_colors.astype(colors.dtype)


# ------------------------------------------------------------------ gravity and boxes


def estimate_up(cam2world: np.ndarray) -> np.ndarray:
    """Unit vector against gravity, from the camera axes (OpenCV: y down, x right).

    Gravity is the mean camera "down" (y) axis, but phones are usually tilted down while
    filming, which biases that mean (by ~7 degrees on scenediff_living_room_1). An upright
    phone keeps its x axis horizontal, so the components of the mean y axis along the
    directions spanned by the camera x axes are removed first (the second direction only
    if the camera also turned sideways). Same idea as `scripts/plot_align.py`.
    """
    _, s, vt = np.linalg.svd(cam2world[:, :3, 0].astype(np.float64))
    horizontal = vt[:2] if s[1] > MIN_YAW_SPREAD * s[0] else vt[:1]
    down = cam2world[:, :3, 1].astype(np.float64).mean(0)
    down -= horizontal.T @ (horizontal @ down)
    norm = np.linalg.norm(down)
    return -down / norm if norm > 1e-6 else np.array([0.0, -1.0, 0.0])


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation (3, 3) that maps the direction `a` onto the direction `b`."""
    a = np.asarray(a, np.float64) / np.linalg.norm(a)
    b = np.asarray(b, np.float64) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:  # opposite: half a turn about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def rotation_y(yaw_deg: float) -> np.ndarray:
    """Rotation about +y: x -> (cos, 0, -sin), z -> (sin, 0, cos) (three.js `rotation.y`)."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def floor_obb(points: np.ndarray) -> dict[str, Any]:
    """Box with a vertical (y) axis around `points` (M, 3), for a levelled world frame.

    The minimum-area rectangle of the (x, z) projection (rotating calipers over the convex
    hull) gives the footprint, the y range the height. Returns `center` (3,), `extents`
    [w, h, d] (w / d along the box's own x / z axis, h vertical) and `yaw_deg` in
    (-45, 45]: the box is `rotation_y(yaw_deg)` applied to an axis-aligned box.
    """
    from scipy.spatial import ConvexHull, QhullError

    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return {"center": [0.0, 0.0, 0.0], "extents": [0.0, 0.0, 0.0], "yaw_deg": 0.0}
    xz = np.unique(pts[:, [0, 2]], axis=0)
    hull = xz
    if len(xz) >= 3:
        with contextlib.suppress(QhullError):  # collinear footprint: keep all points
            hull = xz[ConvexHull(xz).vertices]
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    angles = np.concatenate([[0.0], np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2)])

    def spans(phi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        axes = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
        proj = hull @ axes.T  # coordinates along the box x / z axes
        return proj.min(0), proj.max(0), axes

    areas = [np.prod(hi - lo) for lo, hi, _ in map(spans, angles)]
    phi = float(angles[int(np.argmin(areas))])
    if phi >= np.pi / 4:  # same rectangle with the axes swapped: yaw in (-45, 45]
        phi -= np.pi / 2
    lo, hi, axes = spans(phi)
    mid = axes.T @ ((lo + hi) / 2)
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    yaw = -np.degrees(phi)  # box x axis = (cos phi, 0, sin phi) = rotation_y(-phi) x
    return {"center": np.round([mid[0], (y0 + y1) / 2, mid[1]], 4).tolist(),
            "extents": np.round([hi[0] - lo[0], y1 - y0, hi[1] - lo[1]], 4).tolist(),
            "yaw_deg": round(float(yaw), 2) + 0.0}  # + 0.0: no "-0.0"


def obb_corners(obb: dict[str, Any]) -> np.ndarray:
    """The 8 corners (8, 3) of a `floor_obb` box."""
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    local = signs * np.asarray(obb["extents"], np.float64) / 2
    return local @ rotation_y(obb["yaw_deg"]).T + np.asarray(obb["center"], np.float64)


def obb_footprint(obb: dict[str, Any]) -> np.ndarray:
    """The 4 floor-plane corners (x, z) (4, 2) of a `floor_obb` box, in order around it."""
    w, _, d = np.asarray(obb["extents"], np.float64) / 2
    local = np.array([[-w, 0.0, -d], [-w, 0.0, d], [w, 0.0, d], [w, 0.0, -d]])
    return (local @ rotation_y(obb["yaw_deg"]).T)[:, [0, 2]] + \
        np.asarray(obb["center"], np.float64)[[0, 2]]


def convex_polygon_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Distance between two convex polygons (K, 2) with their corners in order (0 if they
    overlap): separating-axis test on the edge normals, else the closest corner-edge pair."""
    def edges(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return poly, np.roll(poly, -1, axis=0) - poly

    separated = False
    for poly in (P, Q):
        _, e = edges(poly)
        for n in np.c_[-e[:, 1], e[:, 0]]:
            if n @ n < 1e-18:  # degenerate edge of a flat box
                continue
            p, q = P @ n, Q @ n
            if p.max() < q.min() or q.max() < p.min():
                separated = True
                break
        if separated:
            break
    if not separated:
        return 0.0
    best = np.inf
    for points, poly in ((P, Q), (Q, P)):
        starts, e = edges(poly)
        for a, ab in zip(starts, e):
            t = np.clip((points - a) @ ab / max(float(ab @ ab), 1e-18), 0.0, 1.0)
            best = min(best, float(np.linalg.norm(points - (a + t[:, None] * ab), axis=1).min()))
    return best


def obb_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Euclidean gap between two `floor_obb` boxes (0 if they overlap). Both are vertical
    prisms, so the gap of their y ranges and the distance of their footprints combine like
    the legs of a right triangle."""
    def y_range(obb: dict[str, Any]) -> tuple[float, float]:
        y, h = float(obb["center"][1]), float(obb["extents"][1]) / 2
        return y - h, y + h

    (a0, a1), (b0, b1) = y_range(a), y_range(b)
    dy = max(0.0, b0 - a1, a0 - b1)
    return float(np.hypot(dy, convex_polygon_distance(obb_footprint(a), obb_footprint(b))))
