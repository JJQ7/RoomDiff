"""Tests for roomdiff.geometry (numpy, CPU)."""

import numpy as np
import pytest
import torch
from pi3.utils.geometry import recover_intrinsic_from_rays_d
from scipy.spatial.transform import Rotation

from roomdiff.geometry import (
    project_points,
    se3_inverse,
    sim3_apply_to_poses,
    sim3_matrix,
    transform_points,
    umeyama,
    unproject_depth,
    voxel_downsample,
)

K = np.array([[300.0, 0.0, 159.5], [0.0, 310.0, 121.0], [0.0, 0.0, 1.0]])


def random_pose(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=seed).as_matrix()
    T[:3, 3] = rng.normal(size=3)
    return T


def test_se3_inverse_single_and_batched():
    T = np.stack([random_pose(s) for s in range(4)])
    assert np.allclose(se3_inverse(T) @ T, np.eye(4), atol=1e-12)
    assert np.allclose(se3_inverse(T[0]), np.linalg.inv(T[0]))


def test_unproject_project_roundtrip():
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 6.0, (240, 320)).astype(np.float32)
    cam2world = random_pose(1)
    world = unproject_depth(depth, K, cam2world)
    assert world.shape == (240, 320, 3) and world.dtype == np.float32
    u, v, z = project_points(world.astype(np.float64), K, se3_inverse(cam2world))
    rows, cols = np.mgrid[0:240, 0:320]
    assert np.allclose(u, cols, atol=1e-3) and np.allclose(v, rows, atol=1e-3)
    assert np.allclose(z, depth, rtol=1e-5)


def test_project_points_behind_camera_is_nan():
    u, v, z = project_points(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 2.0]]), K, np.eye(4))
    assert np.isnan(u[0]) and np.isnan(v[0]) and z[0] == -1.0
    assert (u[1], v[1]) == (K[0, 2], K[1, 2])


@pytest.mark.parametrize("with_scale", [True, False])
def test_umeyama_recovers_similarity(with_scale):
    rng = np.random.default_rng(2)
    src = rng.normal(size=(50, 3))
    s_true = 1.7 if with_scale else 1.0
    R_true = Rotation.random(random_state=3).as_matrix()
    t_true = np.array([0.3, -2.0, 5.0])
    dst = s_true * src @ R_true.T + t_true
    s, R, t = umeyama(src, dst, with_scale=with_scale)
    assert s == pytest.approx(s_true) and np.allclose(R, R_true) and np.allclose(t, t_true)
    assert np.linalg.det(R) == pytest.approx(1.0)
    # small noise: still close
    s, R, t = umeyama(src, dst + rng.normal(scale=1e-3, size=dst.shape), with_scale)
    assert s == pytest.approx(s_true, rel=1e-2) and np.allclose(R, R_true, atol=1e-2)


def test_umeyama_never_returns_a_reflection():
    rng = np.random.default_rng(4)
    src = rng.normal(size=(20, 3))
    dst = src * np.array([1.0, 1.0, -1.0])  # mirror image: best proper rotation instead
    _, R, _ = umeyama(src, dst)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_sim3_on_cameras_matches_sim3_on_points():
    """Moving a camera by a Sim(3) and scaling its depth moves its points the same way."""
    rng = np.random.default_rng(5)
    depth = rng.uniform(1.0, 3.0, (24, 32)).astype(np.float32)
    cam2world = random_pose(6)
    M = sim3_matrix(0.6, Rotation.random(random_state=7).as_matrix(), [1.0, 2.0, 3.0])
    moved, s = sim3_apply_to_poses(M, cam2world[None])
    assert s == pytest.approx(0.6)
    assert np.allclose(moved[0, :3, :3] @ moved[0, :3, :3].T, np.eye(3))  # still a rotation
    expected = transform_points(M, unproject_depth(depth, K, cam2world))
    assert np.allclose(unproject_depth(depth * s, K, moved[0]), expected, atol=1e-4)


def test_pi3_intrinsics_from_rays_use_our_pixel_convention():
    """K recovered by Pi3 from unit rays unprojects depth to the same camera points."""
    h, w = 42, 56
    K_true = np.array([[50.0, 0.0, 27.0], [0.0, 52.0, 21.5], [0.0, 0.0, 1.0]])
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    xy1 = np.stack([(u - K_true[0, 2]) / K_true[0, 0], (v - K_true[1, 2]) / K_true[1, 1],
                    np.ones_like(u)], -1)
    rays = xy1 / np.linalg.norm(xy1, axis=-1, keepdims=True)
    K_rec = recover_intrinsic_from_rays_d(torch.from_numpy(rays)[None])[0].numpy()
    assert np.allclose(K_rec, K_true, atol=1e-6)

    depth = np.random.default_rng(8).uniform(1, 4, (h, w)).astype(np.float32)
    local_points = xy1 * depth[..., None]  # what Pi3X predicts: (x/z, y/z, 1) * z
    assert np.allclose(unproject_depth(depth, K_rec, np.eye(4)), local_points, atol=1e-4)


def test_voxel_downsample_averages_points_and_colours():
    points = np.array([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [0.51, 0.0, 0.0],
                       [-0.02, 0.0, 0.0]])
    colors = np.array([[0, 0, 0], [100, 50, 3], [10, 20, 30], [7, 7, 7]], np.uint8)
    out, out_colors = voxel_downsample(points, colors, 0.1)
    assert out.dtype == np.float32 and out_colors.dtype == np.uint8 and len(out) == 3
    order = np.lexsort(out.T[::-1])
    assert np.allclose(out[order], [[-0.02, 0, 0], [0.02, 0.02, 0.02], [0.51, 0, 0]])
    assert out_colors[order].tolist() == [[7, 7, 7], [50, 25, 2], [10, 20, 30]]
    alone, none = voxel_downsample(points, None, 1.0)
    assert none is None and len(alone) == 2  # x = -0.02 lies in the cell below 0
