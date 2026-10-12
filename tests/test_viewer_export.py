"""Tests for the static web viewer export on a synthetic recon.npz (CPU only)."""

import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData
from scipy.spatial.transform import Rotation

from roomdiff.recon import load_recon, recording_points, write_recon
from roomdiff.viewer_export import (
    APP_FILES,
    START_VOXEL,
    VIEWER_SRC,
    changes_from_candidates,
    downsample_to_target,
    estimate_up,
    export_viewer,
    main,
)

H, W = 42, 56
K = np.array([[40.0, 0.0, 27.5], [0.0, 40.0, 20.5], [0.0, 0.0, 1.0]], np.float32)
UP = np.array([0.0, 0.0, 1.0])  # the synthetic world is z-up
# OpenCV camera looking along world +x: columns x = -y_w (right), y = -z_w (down), z = +x_w
LEVEL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def upright_camera(yaw_deg: float, pitch_deg: float = 30.0) -> np.ndarray:
    """cam2world of an upright phone (x axis horizontal) 1.5 m above the floor, turned by
    `yaw_deg` about the vertical and tilted down by `pitch_deg`."""
    T = np.eye(4)
    T[:3, :3] = (Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ LEVEL
                 @ Rotation.from_euler("x", -pitch_deg, degrees=True).as_matrix())
    T[:3, 3] = [0.0, 0.0, 1.5]
    return T


def make_run(root: Path, frames_per_recording: int = 4) -> None:
    """Write `<root>/recon/recon.npz` with random depth seen by down-tilted cameras."""
    rng = np.random.default_rng(0)
    n = 2 * frames_per_recording
    recon = {
        "rec": np.repeat([0, 1], frames_per_recording),
        "files": np.array([f"frame_{i:05d}.jpg" for i in range(n)]),
        "images": rng.integers(0, 256, (n, H, W, 3), dtype=np.uint8),
        "cam2world": np.stack([upright_camera(yaw) for yaw in np.linspace(0, 120, n)]),
        "K": np.repeat(K[None], n, 0),
        "depth": rng.uniform(1.5, 2.5, (n, H, W)),
        "conf": np.full((n, H, W), 0.9),
        "valid": rng.uniform(size=(n, H, W)) > 0.2,
        "metric_scale": 1.0,
    }
    (root / "recon").mkdir(parents=True)
    write_recon(root / "recon" / "recon.npz", recon)


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(path))["vertex"]
    assert [p.name for p in vertex.properties] == ["x", "y", "z", "red", "green", "blue"]
    assert vertex["x"].dtype == np.float32 and vertex["red"].dtype == np.uint8
    return (np.stack([vertex["x"], vertex["y"], vertex["z"]], 1),
            np.stack([vertex["red"], vertex["green"], vertex["blue"]], 1))


def plane_cloud(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    points = np.c_[rng.uniform(-1, 1, (n, 2)), np.zeros(n)].astype(np.float32)
    return points, rng.integers(0, 256, (n, 3), dtype=np.uint8)


# ----------------------------------------------------------------------- export


def test_export_writes_plys_manifest_and_app(tmp_path):
    out = tmp_path / "living_room"
    make_run(out)
    viewer = export_viewer(out, target_points=10**6)
    assert viewer == out / "viewer"

    recon = load_recon(out / "recon" / "recon.npz")
    clouds = [recording_points(recon, r) for r in (0, 1)]
    expected, voxel = downsample_to_target(clouds, 10**6)
    assert voxel == START_VOXEL  # the target is not binding
    for name, (points, colors) in zip(("before", "after"), expected):
        ply_points, ply_colors = read_ply(viewer / f"{name}.ply")
        assert np.array_equal(ply_points, points) and np.array_equal(ply_colors, colors)

    scene = json.loads((viewer / "scene.json").read_text())
    assert scene["name"] == "living_room" and scene["units"] == "metres (approx.)"
    assert (scene["before_ply"], scene["after_ply"], scene["changes"]) == (
        "before.ply", "after.ply", "changes.json")
    stats = scene["stats"]
    assert [stats["points_before"], stats["points_after"]] == [len(p) for p, _ in expected]
    lo, hi = np.percentile(np.concatenate([p for p, _ in expected]), [1, 99], axis=0)
    assert np.allclose(stats["extent_m"], hi - lo, atol=1e-3)
    assert np.allclose(scene["bbox_m"], [lo, hi], atol=1e-3)
    assert np.allclose(scene["up_axis"], UP, atol=1e-3)  # despite the 30 degree tilt
    assert abs(np.dot(scene["view_dir"], UP)) < 1e-3

    changes = json.loads((viewer / "changes.json").read_text())
    assert changes["changes"] == [] and changes["summary"]
    for file in APP_FILES:
        assert (viewer / file).read_bytes() == (VIEWER_SRC / file).read_bytes()


def test_export_respects_target_points(tmp_path):
    make_run(tmp_path)
    main([str(tmp_path), "--target-points", "500"])
    scene = json.loads((tmp_path / "viewer" / "scene.json").read_text())
    for name in ("before", "after"):
        points, _ = read_ply(tmp_path / "viewer" / f"{name}.ply")
        assert 0 < len(points) <= 500
        assert len(points) == scene["stats"][f"points_{name}"]
    assert scene["stats"]["voxel_m"] > START_VOXEL


def test_export_copies_changes_and_crops(tmp_path):
    make_run(tmp_path)
    (tmp_path / "crops").mkdir()
    (tmp_path / "crops" / "1_before.jpg").write_bytes(b"jpeg")
    changes = {
        "scene": {"before": "a.mov", "after": "b.mov", "units": "metres (approx.)"},
        "changes": [{"id": 1, "type": "removed", "label": "lamp", "confidence": 0.7,
                     "before": {"centroid": [0, 0, 0], "aabb": [[0, 0, 0], [1, 1, 1]]},
                     "crops": {"before": "crops/1_before.jpg", "after": "crops/1_after.jpg"}}],
        "summary": "1 change",
    }
    (tmp_path / "changes.json").write_text(json.dumps(changes))
    viewer = export_viewer(tmp_path, target_points=1000)
    copied = json.loads((viewer / "changes.json").read_text())
    assert copied["changes"][0]["crops"] == {"before": "crops/1_before.jpg", "after": None}
    assert (viewer / "crops" / "1_before.jpg").read_bytes() == b"jpeg"
    copied["changes"][0]["crops"] = changes["changes"][0]["crops"]
    assert copied == changes  # nothing else changed


def test_export_falls_back_to_visibility_candidates(tmp_path):
    make_run(tmp_path)
    debug = tmp_path / "visibility" / "debug"
    debug.mkdir(parents=True)
    (debug / "blob_1_appeared_after_0.jpg").write_bytes(b"jpeg")
    blob = {"id": 1, "type": "appeared", "voxels": 40, "centroid": [0.5, 0.5, 0.5],
            "aabb": [[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]],
            "evidence": {"occ_points": 300, "observing_views": 7},
            "views": {"before": [], "after": [{"crop": "debug/blob_1_appeared_after_0.jpg"}]}}
    (tmp_path / "visibility" / "candidates.json").write_text(json.dumps({"candidates": [blob]}))
    viewer = export_viewer(tmp_path, target_points=1000)
    change = json.loads((viewer / "changes.json").read_text())["changes"][0]
    assert change["crops"] == {"after": "crops/blob_1_appeared_after_0.jpg"}
    assert (viewer / change["crops"]["after"]).is_file()


# ---------------------------------------------------------------------- helpers


@pytest.mark.parametrize("target", [1, 100, 2000])
def test_downsample_to_target(target):
    clouds = [plane_cloud(20000, 0), plane_cloud(5000, 1)]
    out, voxel = downsample_to_target(clouds, target)
    sizes = [len(p) for p, _ in out]
    assert max(sizes) <= target and max(sizes) > target / 2  # fits without overshooting
    assert voxel > START_VOXEL and all(c.dtype == np.uint8 for _, c in out)
    for (points, _), (small, _) in zip(clouds, out):  # averages stay inside the cloud
        assert np.all(small.min(0) >= points.min(0) - 1e-6)
        assert np.all(small.max(0) <= points.max(0) + 1e-6)
    with pytest.raises(ValueError, match="target_points"):
        downsample_to_target(clouds, 0)


def test_estimate_up_is_robust_to_camera_tilt():
    cameras = np.stack([upright_camera(yaw, pitch) for yaw, pitch in
                        zip(np.linspace(-40, 40, 9), [20, 30, 35, 25, 40, 30, 20, 35, 30])])
    naive = -cameras[:, :3, 1].mean(0)
    naive /= np.linalg.norm(naive)
    assert np.degrees(np.arccos(naive @ UP)) > 20  # the plain mean is biased by the tilt
    assert np.allclose(estimate_up(cameras), UP, atol=1e-9)
    # camera only tilting, never turning: still a unit vector, perpendicular to the x axes
    tilting = np.stack([upright_camera(10, pitch) for pitch in (0, 20, 40)])
    up = estimate_up(tilting)
    assert np.isclose(np.linalg.norm(up), 1) and abs(up @ tilting[0, :3, 0]) < 1e-9


def test_changes_from_candidates_maps_types():
    candidates = {"candidates": [
        {"id": i + 1, "type": kind, "centroid": [0, 0, 0], "aabb": [[0, 0, 0], [0.5, 0.2, 0.1]],
         "evidence": {"observing_views": 3}}
        for i, kind in enumerate(["disappeared", "appeared", "unverified_before",
                                  "unverified_after", "something_new"])
    ]}
    doc = changes_from_candidates(candidates, {"units": "m"})
    assert [(c["type"], "before" in c, "after" in c) for c in doc["changes"]] == [
        ("removed", True, False), ("added", False, True),
        ("unverified", True, False), ("unverified", False, True)]
    first = doc["changes"][0]
    assert first["label"] == "region" and first["confidence"] is None and first["crops"] == {}
    assert "0.50 x 0.20 x 0.10 m" in first["description"] and "3 views" in first["description"]
    assert doc["scene"] == {"units": "m"}
    assert doc["summary"].startswith("4 candidate regions") and "1 removed, 1 added, 2 unverified" in doc["summary"]
