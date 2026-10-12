"""Small numpy geometry helpers (CPU only).

Conventions: OpenCV cameras (x right, y down, z forward); `cam2world` / `world2cam` are
4x4 rigid transforms; pixel (u, v) = (column, row) with pixel centres at integer
coordinates, the convention of `pi3.utils.geometry.recover_intrinsic_from_rays_d`, so an
intrinsic matrix recovered from Pi3X rays can be used here directly.
"""

from __future__ import annotations

import numpy as np


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
