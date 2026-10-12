"""Tests for M2 recon and the Rerun export on synthetic data (no model, no GPU).

A fake Pi3X "renders" a box-shaped room for keyframes whose identity is encoded in their
colour and returns the result in a random Sim(3) gauge per call, like a real
feed-forward model whose output frame depends on its input set.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from typer.testing import CliRunner

from roomdiff import recon as recon_mod
from roomdiff.cli import app
from roomdiff.config import load_config
from roomdiff.geometry import sim3_apply_to_poses, sim3_matrix, umeyama
from roomdiff.recon import (
    alignment_diagnostics,
    anchor_indices,
    fit_floor,
    level_transform,
    load_recon,
    observable_by,
    preprocess_images,
    run_recon,
    set_head_chunk,
    target_size,
    verdict_line,
    write_ply,
    write_recon,
)
from roomdiff.viz_rerun import log_recon

H, W = 84, 112  # multiples of 14
K_TRUE = np.array([[80.0, 0.0, 55.5], [0.0, 80.0, 41.5], [0.0, 0.0, 1.0]], np.float32)
ROOM_LO, ROOM_HI = np.array([-2.0, -1.2, -2.5]), np.array([2.0, 1.3, 2.5])
YAWS = {"before": np.linspace(0, 180, 8), "after": np.linspace(20, 200, 8)}


# ------------------------------------------------------------------ synthetic room


def yaw_pose(yaw_deg: float) -> np.ndarray:
    """Camera 0.5 m from the room centre, looking outwards-ish, rotated about y."""
    a = np.radians(yaw_deg)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("y", a).as_matrix()
    T[:3, 3] = [0.5 * np.cos(a), 0.1 * np.sin(3 * a), 0.5 * np.sin(a)]
    return T


GT_POSES = np.stack([yaw_pose(y) for name in ("before", "after") for y in YAWS[name]])


def render_depth(cam2world: np.ndarray) -> np.ndarray:
    """z-depth of the inside of the box room seen from `cam2world`."""
    v, u = np.mgrid[0:H, 0:W]
    d_cam = np.stack([(u - K_TRUE[0, 2]) / K_TRUE[0, 0], (v - K_TRUE[1, 2]) / K_TRUE[1, 1],
                      np.ones((H, W))], -1)
    d = d_cam @ cam2world[:3, :3].T
    c = cam2world[:3, 3]
    with np.errstate(divide="ignore"):
        exits = np.maximum((ROOM_LO - c) / d, (ROOM_HI - c) / d)
    return exits.min(-1).astype(np.float32)  # d_cam has z = 1, so ray length = z-depth


def colour_of(frame_id: int) -> tuple[int, int, int]:
    return 12 * frame_id + 6, 128, 128


def fake_run_pi3x(images, model, conf_threshold=0.25, edge_rtol=0.03):
    """Stand-in for `run_pi3x`: exact geometry in a new random Sim(3) gauge per call."""
    ids = np.rint((images[:, 0].mean(dim=(1, 2)).numpy() * 255 - 6) / 12).astype(int)
    fake_run_pi3x.calls += 1
    rng = np.random.default_rng(fake_run_pi3x.calls)
    s = rng.uniform(0.5, 2.0)
    M = sim3_matrix(s, Rotation.random(random_state=rng).as_matrix(), rng.normal(size=3))
    cam2world, _ = sim3_apply_to_poses(M, GT_POSES[ids])
    n = len(ids)
    return {
        "cam2world": cam2world.astype(np.float32),
        "K": np.repeat(K_TRUE[None], n, 0),
        "depth": s * np.stack([render_depth(GT_POSES[i]) for i in ids]),
        "conf": np.full((n, H, W), 0.9, np.float32),
        "valid": np.ones((n, H, W), bool),
        "metric_scale": s,
    }


fake_run_pi3x.calls = 0


@pytest.fixture
def scene_dir(tmp_path, monkeypatch) -> Path:
    """Output dir with flat-colour keyframes + index.json; Pi3X replaced by the fake."""
    out = tmp_path / "out"
    index = {}
    frame_id = 0
    for name in ("before", "after"):
        folder = out / "keyframes" / name
        folder.mkdir(parents=True)
        frames = []
        for i in range(len(YAWS[name])):
            file = f"{name}_{i:05d}.jpg"
            Image.new("RGB", (W, H), colour_of(frame_id)).save(folder / file, quality=95)
            frames.append({"file": file})
            frame_id += 1
        index[name] = {"dir": name, "frames": frames}
    (out / "keyframes" / "index.json").write_text(json.dumps(index))
    monkeypatch.setattr(recon_mod, "load_model", lambda name, **kwargs: object())
    monkeypatch.setattr(recon_mod, "run_pi3x", fake_run_pi3x)
    return out


def test_fake_room_depth_is_consistent():
    """Sanity check of the test fixture itself: every unprojected point is on a wall."""
    from roomdiff.geometry import unproject_depth

    pts = unproject_depth(render_depth(GT_POSES[3]), K_TRUE, GT_POSES[3]).reshape(-1, 3)
    on_wall = np.isclose(pts, ROOM_LO, atol=1e-4) | np.isclose(pts, ROOM_HI, atol=1e-4)
    assert on_wall.any(axis=1).all()


# ------------------------------------------------------------------ preprocessing


def test_target_size_follows_pi3_rule():
    assert target_size(1024, 576) == (672, 378)
    assert target_size(576, 1024) == (378, 672)
    assert target_size(W, H, W * H) == (W, H)


def test_preprocess_matches_pi3_loader(tmp_path):
    from pi3.utils.basic import load_images_as_tensor

    rng = np.random.default_rng(0)
    for i in range(2):
        array = rng.integers(0, 256, (150, 200, 3), dtype=np.uint8)
        Image.fromarray(array).save(tmp_path / f"im_{i}.png")
    files = sorted(tmp_path.glob("*.png"))
    tensor, images, scales = preprocess_images(files, pixel_limit=60000)
    reference = load_images_as_tensor(str(tmp_path), PIXEL_LIMIT=60000, verbose=False)
    assert torch.equal(tensor, reference)
    assert images.shape == (2, *tensor.shape[2:], 3) and images.dtype == np.uint8
    assert np.allclose(scales, [tensor.shape[3] / 200, tensor.shape[2] / 150])


def test_preprocess_rejects_mixed_orientation(tmp_path):
    Image.new("RGB", (1024, 576)).save(tmp_path / "a.jpg")
    Image.new("RGB", (576, 1024)).save(tmp_path / "b.jpg")
    with pytest.raises(ValueError, match="same orientation"):
        preprocess_images([tmp_path / "a.jpg", tmp_path / "b.jpg"])


def test_preprocess_resizes_same_aspect_to_one_size(tmp_path):
    Image.new("RGB", (1024, 576)).save(tmp_path / "a.jpg")
    Image.new("RGB", (640, 360)).save(tmp_path / "b.jpg")
    tensor, _, scales = preprocess_images([tmp_path / "a.jpg", tmp_path / "b.jpg"])
    assert tensor.shape == (2, 3, 378, 672)
    assert np.allclose(scales, [[672 / 1024, 378 / 576], [672 / 640, 378 / 360]])


def test_anchor_indices():
    assert anchor_indices(48, 8).tolist() == [0, 7, 13, 20, 27, 34, 40, 47]
    assert anchor_indices(3, 8).tolist() == [0, 1, 2]


def test_set_head_chunk_overrides_the_default_chunk_size():
    import inspect

    from pi3.models.pi3x import Pi3X

    # Pi3X.forward_head relies on this default; fail loudly if upstream renames it
    assert inspect.signature(Pi3X._chunked_conv_head).parameters["chunk_size"].default == 64

    class Heads(torch.nn.Module):
        def _chunked_conv_head(self, head, feat, patch_h, patch_w, chunk_size=64):
            return [len(c) for c in feat.split(chunk_size)]

    model = Heads()
    set_head_chunk(model, 16)
    assert model._chunked_conv_head(None, torch.zeros(40), 1, 1) == [16, 16, 8]
    assert Heads()._chunked_conv_head(None, torch.zeros(40), 1, 1) == [40]  # class intact
    with pytest.raises(ValueError, match="head_chunk"):
        set_head_chunk(model, 0)


# ---------------------------------------------------------------------------- I/O


def synthetic_recon(n: int = 3) -> dict:
    rng = np.random.default_rng(0)
    return {
        "rec": np.array([0] * (n - 1) + [1]),
        "files": np.array([f"before_{i:05d}.jpg" for i in range(n - 1)] + ["after_00000.jpg"]),
        "images": rng.integers(0, 256, (n, H, W, 3), dtype=np.uint8),
        "cam2world": GT_POSES[:n],
        "K": np.repeat(K_TRUE[None], n, 0),
        "depth": np.stack([render_depth(T) for T in GT_POSES[:n]]),
        "conf": rng.uniform(0, 1, (n, H, W)),
        "valid": rng.uniform(0, 1, (n, H, W)) > 0.3,
        "metric_scale": 1.25,
    }


def test_write_and_load_recon_roundtrip(tmp_path):
    data = synthetic_recon()
    write_recon(tmp_path / "recon.npz", data)
    loaded = load_recon(tmp_path / "recon.npz")
    dtypes = {"rec": np.uint8, "images": np.uint8, "cam2world": np.float32,
              "K": np.float32, "depth": np.float32, "conf": np.float16, "valid": np.bool_}
    for key, dtype in dtypes.items():
        assert loaded[key].dtype == dtype, key
        assert np.allclose(loaded[key], data[key], atol=1e-3), key
    assert loaded["files"].tolist() == data["files"].tolist()
    assert loaded["metric_scale"] == 1.25 and isinstance(loaded["metric_scale"], float)


def test_write_recon_rejects_inconsistent_shapes(tmp_path):
    data = synthetic_recon()
    data["K"] = data["K"][:2]
    with pytest.raises(ValueError, match="'K'"):
        write_recon(tmp_path / "recon.npz", data)


def test_write_ply_roundtrip(tmp_path):
    points = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    colors = np.arange(30, dtype=np.uint8).reshape(10, 3)
    write_ply(tmp_path / "c.ply", points, colors)
    vertex = PlyData.read(str(tmp_path / "c.ply"))["vertex"]
    assert np.allclose(np.stack([vertex["x"], vertex["y"], vertex["z"]], 1), points)
    assert np.stack([vertex["red"], vertex["green"], vertex["blue"]], 1).tolist() == colors.tolist()


def overlap_recon(after_views: list[int]) -> dict:
    """Recon with before = GT views 0, 1 and after = `after_views`, exact depth."""
    views = [0, 1, *after_views]
    data = synthetic_recon(len(views))
    data["rec"] = np.array([0, 0] + [1] * len(after_views))
    data["cam2world"] = GT_POSES[views].copy()
    data["depth"] = np.stack([render_depth(T) for T in data["cam2world"]])
    data["valid"][:] = True
    return data


def test_observable_by():
    data = overlap_recon([0])
    data["cam2world"][:] = np.eye(4)
    data["depth"][:] = 2.0  # a wall 2 m in front of every camera
    data["valid"][0, 0, :] = False  # top image row of before camera 0 is invalid
    data["valid"][1] = False  # before camera 1 sees nothing valid, so only camera 0 counts
    v_top = -K_TRUE[1, 2] / K_TRUE[1, 1]  # direction through pixel row 0
    points = np.array([
        [0, 0, 1.0],  # in front of the wall: seen empty -> observable
        [0, 0, 2.05],  # on the wall within tolerance
        [0, 0, 3.0],  # behind the wall: hidden
        [0, 0, -1.0],  # behind the camera
        [10, 0, 1.0],  # outside the image
        [0, v_top, 1.0],  # projects into an invalid pixel
    ])
    assert observable_by(points, data, 0).tolist() == [True, True, False, False, False, False]
    # the after camera (all pixels valid) does see the last point
    assert observable_by(points, data, 1).tolist() == [True, True, False, False, False, True]


def test_alignment_diagnostics_detect_a_shift():
    data = overlap_recon([0, 1])  # after = same views: perfect overlap
    diag, clouds = alignment_diagnostics(data, voxel=0.02)
    assert diag["verdict"] == "alignment OK" and diag["verdict_reason"] == ""
    covis = diag["nn_covisible"]["before_to_after"]
    assert covis["median_m"] < 1e-6 and covis["fraction"] > 0.99
    assert diag["before"]["camera_path_m"] == pytest.approx(
        np.linalg.norm(GT_POSES[1, :3, 3] - GT_POSES[0, :3, 3]))
    assert len(clouds) == 2 and clouds[0][1].dtype == np.uint8
    extent = np.array(diag["scene_extent_m"])
    assert np.all(extent > 0) and np.all(extent <= ROOM_HI - ROOM_LO + 1e-6)
    assert verdict_line(diag).startswith("alignment OK (before->after: 100% co-visible")

    data["cam2world"][2:, 0, 3] += 0.3  # after shifted by 30 cm
    diag, _ = alignment_diagnostics(data, voxel=0.02)
    assert diag["verdict"] == "alignment WEAK" and "within 10 cm" in diag["verdict_reason"]
    assert diag["nn_covisible"]["before_to_after"]["frac_lt_10cm"] < 0.9


def test_alignment_verdict_ignores_areas_seen_by_one_video_only():
    data = overlap_recon([0, 1, 7, 15])  # after also films the opposite side of the room
    diag, _ = alignment_diagnostics(data, voxel=0.02)
    assert diag["nn_all"]["after_to_before"]["frac_lt_10cm"] < 0.8  # all points: looks bad
    assert diag["nn_covisible"]["after_to_before"]["fraction"] < 0.7
    assert diag["nn_covisible"]["after_to_before"]["median_m"] < 1e-6
    assert diag["verdict"] == "alignment OK"


def test_alignment_verdict_without_overlap():
    data = overlap_recon([7, 15])  # after films only what before never saw
    diag, _ = alignment_diagnostics(data, voxel=0.02)
    assert diag["verdict"] == "alignment WEAK"
    assert diag["verdict_reason"] == "little co-visible overlap"


# ------------------------------------------------------------------- full stage


def assert_matches_ground_truth(recon: dict, tol: float) -> None:
    """All cameras of both recordings are one Sim(3) away from the ground truth."""
    s, R, t = umeyama(GT_POSES[:, :3, 3], recon["cam2world"][:, :3, 3])
    expected = s * GT_POSES[:, :3, 3] @ R.T + t
    assert np.abs(recon["cam2world"][:, :3, 3] - expected).max() < tol * s
    rot_err = Rotation.from_matrix(R @ GT_POSES[:, :3, :3]).inv() * Rotation.from_matrix(
        recon["cam2world"][:, :3, :3].astype(np.float64))
    assert np.degrees(rot_err.magnitude()).max() < 0.5
    gt_depth = np.stack([render_depth(T) for T in GT_POSES])
    assert np.allclose(recon["depth"], s * gt_depth, rtol=1e-3)


def test_run_recon_joint(scene_dir, capsys):
    meta = run_recon(scene_dir, load_config(overrides=["recon.pixel_limit=9408"]))
    recon = load_recon(scene_dir / "recon" / "recon.npz")
    assert recon["rec"].tolist() == [0] * 8 + [1] * 8
    assert recon["files"][8] == "after_00000.jpg" and recon["images"].shape == (16, H, W, 3)
    assert_matches_ground_truth(recon, tol=1e-4)

    assert meta == json.loads((scene_dir / "recon" / "meta.json").read_text())
    assert meta["mode"] == "joint" and meta["resolution"] == [W, H]
    assert meta["frames"] == {"before": 8, "after": 8} and meta["peak_vram_gb"] is None
    assert meta["diagnostics"]["verdict"] == "alignment OK"
    assert "alignment OK" in capsys.readouterr().out
    for name in ("before", "after"):
        assert len(PlyData.read(str(scene_dir / "recon" / f"{name}.ply"))["vertex"]) > 1000


def floor_tilt_and_height(recon: dict) -> tuple[float, float]:
    """Angle (deg) between the ground-truth up and +Y, and the largest |y| of the floor
    corners, in the frame of `recon` (GT cameras are upright: world +y is down)."""
    s, R, t = umeyama(GT_POSES[:, :3, 3], recon["cam2world"][:, :3, 3])
    up = R @ np.array([0.0, -1.0, 0.0])
    corners = np.array([[x, ROOM_HI[1], z] for x in ROOM_LO[[0]].tolist() + ROOM_HI[[0]].tolist()
                        for z in (ROOM_LO[2], ROOM_HI[2])])
    floor = s * corners @ R.T + t
    return float(np.degrees(np.arccos(np.clip(up[1], -1, 1)))), float(np.abs(floor[:, 1]).max())


def test_run_recon_levels_the_world(scene_dir):
    """The fake model returns the room in a random (tilted) gauge; run_recon levels it:
    floor normal along +Y, floor at y = 0, first before camera looking along +Z. The floor
    is only a thin strip along the wall bottoms here (cameras look horizontally from the
    centre), a hard case: plain RANSAC inlier counting tilted this gauge by 1.2 degrees."""
    fake_run_pi3x.calls = 0  # a fixed gauge (scale 0.89)
    meta = run_recon(scene_dir, load_config(overrides=["recon.pixel_limit=9408"]))
    recon = load_recon(scene_dir / "recon" / "recon.npz")
    tilt, height = floor_tilt_and_height(recon)
    assert tilt < 1.0 and height < 0.01
    forward = recon["cam2world"][0, :3, 2]
    assert abs(forward[0]) < 1e-3 and forward[2] > 0.99
    assert meta["levelled"] and meta["level"]["method"] == "ransac floor plane"
    assert meta["floor_inliers_fraction"] > 0.1 and len(meta["up_axis_original"]) == 3
    # the first camera already faces a wall: snapping keeps the frame (within 0.06 deg)
    assert meta["walls_snapped"] and abs(meta["wall_yaw_deg"]) < 0.06
    assert meta["wall_peak_fraction"] > 0.9 and meta["level"]["wall_votes"] > 100

    fake_run_pi3x.calls -= 1  # same gauge, levelling off: the floor stays tilted
    meta = run_recon(scene_dir, load_config(
        overrides=["recon.pixel_limit=9408", "recon.level_world=false"]))
    assert not meta["levelled"] and "level" not in meta
    assert floor_tilt_and_height(load_recon(scene_dir / "recon" / "recon.npz"))[0] > 1.0


def test_floor_plane_corrects_the_camera_up():
    """Rolled cameras bias the camera up estimate by ~8 degrees; the floor fit removes it."""
    poses = []
    for yaw in np.linspace(0, 40, 8):
        T = yaw_pose(yaw)
        T[:3, :3] = T[:3, :3] @ Rotation.from_euler("zx", [8, -15], degrees=True).as_matrix()
        poses.append(T)
    poses = np.stack(poses)
    n = len(poses)
    recon = {"cam2world": poses, "K": np.repeat(K_TRUE[None], n, 0),
             "depth": np.stack([render_depth(T) for T in poses]),
             "valid": np.ones((n, H, W), bool), "rec": np.zeros(n, int)}
    T, info = level_transform(recon)
    assert info["floor_vs_camera_up_deg"] > 5  # the camera estimate alone is off
    up = T[:3, :3] @ np.array([0.0, -1.0, 0.0])
    assert np.degrees(np.arccos(up[1])) < 1.0
    assert abs((T[:3, :3] @ [0.0, ROOM_HI[1], 0.0] + T[:3, 3])[1]) < 0.01
    assert info["floor_inliers_fraction"] > 0.5


def wall_axis_yaws(T: np.ndarray, M: np.ndarray) -> list[float]:
    """Yaw (deg, folded into [-45, 45)) of the room's x and z axes (the wall normals) after
    the ground-truth -> model similarity `M` and the levelling transform `T`."""
    yaws = []
    for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])):
        d = T[:3, :3] @ M[:3, :3] @ axis
        d /= np.linalg.norm(d)
        assert abs(d[1]) < 0.02  # levelled: the wall normals are horizontal
        yaws.append(float((np.degrees(np.arctan2(-d[2], d[0])) + 45) % 90 - 45))
    return yaws


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_snap_to_walls_aligns_a_rotated_room(seed):
    """The first camera looks 20 degrees off the walls, so levelling alone leaves the walls
    at 20 degrees to the axes; snapping turns them onto the axes (within 1 degree)."""
    rng = np.random.default_rng(seed)
    gt = np.stack([yaw_pose(y + 20) for y in YAWS["before"]])
    s = rng.uniform(0.7, 1.5)
    M = sim3_matrix(s, Rotation.random(random_state=rng).as_matrix(), rng.normal(size=3))
    poses, _ = sim3_apply_to_poses(M, gt)
    n = len(poses)
    recon = {"cam2world": poses, "K": np.repeat(K_TRUE[None], n, 0),
             "depth": s * np.stack([render_depth(T) for T in gt]),
             "valid": np.ones((n, H, W), bool), "rec": np.zeros(n, int)}
    plain, info = level_transform(recon)
    assert "wall_yaw_deg" not in info
    assert all(abs(abs(a) - 20) < 1 for a in wall_axis_yaws(plain, M))

    T, info = level_transform(recon, snap_to_walls=True)
    assert info["walls_snapped"] and abs(abs(info["wall_yaw_deg"]) - 20) < 1
    assert info["wall_peak_fraction"] > 0.5 and info["wall_votes"] > 100
    assert all(abs(a) < 1 for a in wall_axis_yaws(T, M))
    assert np.allclose(T[1], plain[1])  # a turn about +Y: heights (floor at y = 0) unchanged

    # a peak below wall_peak_fraction leaves the frame as it is (the yaw is still reported)
    T, info = level_transform(recon, snap_to_walls=True, wall_peak_fraction=1.01)
    assert not info["walls_snapped"] and abs(abs(info["wall_yaw_deg"]) - 20) < 1
    assert np.allclose(T, plain)


def test_fit_floor_ignores_steep_planes():
    rng = np.random.default_rng(0)
    xz = rng.uniform(-1, 1, (4000, 2))
    up = np.array([0.0, 1.0, 0.0])
    ramp = np.c_[xz[:, 0], xz[:, 0], xz[:, 1]]  # 45 degrees from horizontal
    assert fit_floor(ramp, up) is None
    tilted = np.c_[xz[:, 0], 0.05 * xz[:, 0] + rng.normal(0, 0.003, len(xz)), xz[:, 1]]
    clutter = rng.uniform([-1, 0.1, -1], [1, 1.5, 1], (20000, 3))
    normal, offset, inliers = fit_floor(np.vstack([tilted, clutter]), up)
    truth = np.array([-0.05, 1.0, 0.0]) / np.linalg.norm([-0.05, 1.0, 0.0])
    assert np.degrees(np.arccos(normal @ truth)) < 0.5 and abs(offset) < 0.005
    assert inliers > 0.9


def test_run_recon_separate_recovers_the_shared_frame(scene_dir):
    overrides = ["recon.pixel_limit=9408", "recon.mode=separate", "recon.n_anchor=4"]
    meta = run_recon(scene_dir, load_config(overrides=overrides))
    recon = load_recon(scene_dir / "recon" / "recon.npz")
    assert_matches_ground_truth(recon, tol=5e-3)  # ICP may move a few mm on 2 cm voxels
    alignment = meta["alignment"]
    assert alignment["method"] == "separate" and alignment["anchors"] == [4, 4]
    assert alignment["anchor_rmse_before_m"] < 1e-4 and alignment["icp"]["accepted"]
    assert len(alignment["metric_scale_per_pass"]) == 3
    assert meta["diagnostics"]["verdict"] == "alignment OK"


def test_scale_factor_scales_metric_outputs(scene_dir):
    run_recon(scene_dir, load_config(overrides=["recon.pixel_limit=9408"]))
    plain = load_recon(scene_dir / "recon" / "recon.npz")
    fake_run_pi3x.calls -= 1  # replay the same random gauge
    run_recon(scene_dir, load_config(
        overrides=["recon.pixel_limit=9408", "recon.scale_factor=2"]))
    scaled = load_recon(scene_dir / "recon" / "recon.npz")
    assert np.allclose(scaled["depth"], 2 * plain["depth"])
    assert np.allclose(scaled["cam2world"][:, :3, 3], 2 * plain["cam2world"][:, :3, 3])
    assert np.allclose(scaled["cam2world"][:, :3, :3], plain["cam2world"][:, :3, :3])


def test_run_recon_refuses_too_many_frames(scene_dir):
    with pytest.raises(ValueError, match="max_frames"):
        run_recon(scene_dir, load_config(overrides=["recon.max_frames=10"]))


def test_cli_recon_and_viz(scene_dir):
    runner = CliRunner()
    result = runner.invoke(app, ["recon", "--out", str(scene_dir),
                                 "-s", "recon.pixel_limit=9408"])
    assert result.exit_code == 0, result.output
    assert (scene_dir / "recon" / "recon.npz").exists()
    result = runner.invoke(app, ["viz", "--out", str(scene_dir)])
    assert result.exit_code == 0, result.output
    assert (scene_dir / "recon" / "recon.rrd").stat().st_size > 10_000
    assert (scene_dir / "scene.rrd").stat().st_size > 10_000  # no visibility output yet


def test_log_recon_writes_rrd(scene_dir):
    run_recon(scene_dir, load_config(overrides=["recon.pixel_limit=9408"]))
    rrd = log_recon(scene_dir, max_points=5000)
    assert rrd == scene_dir / "recon" / "recon.rrd" and rrd.stat().st_size > 10_000


def test_cli_run_chains_ingest_recon_visibility_viz(tmp_path, monkeypatch):
    import roomdiff.changes
    import roomdiff.ingest
    import roomdiff.instances
    import roomdiff.report
    import roomdiff.semantics
    import roomdiff.viewer_export
    import roomdiff.visibility
    import roomdiff.viz_rerun

    calls = []
    monkeypatch.setattr(roomdiff.ingest, "ingest_pair", lambda *a: calls.append("ingest"))
    monkeypatch.setattr(recon_mod, "run_recon", lambda *a: calls.append("recon"))
    monkeypatch.setattr(roomdiff.visibility, "run_visibility",
                        lambda *a: calls.append("visibility"))
    monkeypatch.setattr(roomdiff.instances, "run_instances", lambda *a: calls.append("instances"))
    monkeypatch.setattr(roomdiff.changes, "run_changes", lambda *a: calls.append("changes"))
    monkeypatch.setattr(roomdiff.semantics, "run_semantics", lambda *a: calls.append("semantics"))
    monkeypatch.setattr(roomdiff.report, "write_report", lambda *a: calls.append("report"))
    monkeypatch.setattr(roomdiff.viewer_export, "export_viewer", lambda *a: calls.append("viewer"))
    for name in ("log_visibility", "log_changes", "log_scene", "log_recon"):
        monkeypatch.setattr(roomdiff.viz_rerun, name, lambda *a, n=name: calls.append(n))
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    args = ["run", "--before", str(video), "--after", str(video), "--out", str(tmp_path)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert calls == ["ingest", "recon", "visibility", "log_visibility", "instances", "changes",
                     "log_changes", "semantics", "report", "log_changes", "log_scene",
                     "log_recon", "viewer"]
    assert "[run] wall time: ingest" in result.output and "visibility" in result.output
