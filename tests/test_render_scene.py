"""Tests for scripts/render_scene.py on a tiny synthetic run folder (CPU only, < 1 s)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageSequence
from plyfile import PlyData, PlyElement

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_scene.py"


@pytest.fixture(scope="module")
def rs():
    spec = importlib.util.spec_from_file_location("render_scene", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_scene"] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertex = np.empty(len(points), [("x", "f4"), ("y", "f4"), ("z", "f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, k in enumerate("xyz"):
        vertex[k] = points[:, i]
    for i, k in enumerate(("red", "green", "blue")):
        vertex[k] = colors[:, i]
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))


def side(center, extents, yaw=0.0):
    return {"centroid": list(center), "obb": {"center": list(center), "extents": list(extents),
                                              "yaw_deg": yaw}}


def make_run(root: Path) -> Path:
    """A 4 x 1.5 x 3 m 'room' of grey points with one change of each type."""
    rng = np.random.default_rng(0)
    run = root / "scenediff_synthetic"
    (run / "viewer").mkdir(parents=True)
    lo, hi = np.array([-2.0, 0.0, -1.5]), np.array([2.0, 1.5, 1.5])
    for i, name in enumerate(("before", "after")):
        points = rng.uniform(lo, hi, (400, 3))
        points[:200, 1] = 0.0  # a floor
        grey = rng.integers(90, 170, (400, 1))
        write_ply(run / "viewer" / f"{name}.ply", points, np.repeat(grey + 10 * i, 3, axis=1))
    (run / "viewer" / "scene.json").write_text(json.dumps({
        "name": "scenediff_synthetic", "up_axis": [0.0, 1.0, 0.0],
        "bbox_m": [lo.tolist(), hi.tolist()]}))
    moved = {"id": 1, "type": "moved", "label": "chair",
             "before": side([-1.2, 0.4, -0.5], [0.5, 0.8, 0.5], 10.0),
             "after": side([-0.2, 0.4, 0.3], [0.5, 0.8, 0.5], 40.0),
             "transform": {"R": np.eye(3).tolist(), "t": [1.0, 0.0, 0.8],
                           "translation_m": 1.28}}
    changes = [
        moved,
        {"id": 2, "type": "removed", "label": "bag", "before": side([1.2, 0.2, -0.8],
                                                                   [0.4, 0.4, 0.3]),
         "after": None},
        {"id": 3, "type": "added", "label": "box", "before": None,
         "after": side([1.0, 0.3, 0.8], [0.4, 0.6, 0.4], -20.0)},
        {"id": 4, "type": "unverified", "label": "shelf", "before": None,
         "after": side([-1.5, 0.7, 1.0], [0.6, 1.4, 0.3])},
    ]
    (run / "changes.json").write_text(json.dumps({"changes": changes}))
    return run


def near(image: Image.Image, color, tol: int = 30) -> int:
    """Number of pixels within `tol` of `color` in every channel."""
    rgb = np.asarray(image.convert("RGB")).astype(int)
    return int((np.abs(rgb - np.array(color)).max(axis=2) <= tol).sum())


def test_render_synthetic_run(rs, tmp_path):
    run = make_run(tmp_path)
    out = tmp_path / "img" / "synthetic"
    assert rs.main([str(run), "--out", str(out), "--width", "300", "--gif"]) == 0

    names = ("overview", "before", "after")
    paths = {k: out.with_name(f"synthetic_{k}.png") for k in names}
    gif = out.with_name("synthetic_beforeafter.gif")
    for path in [*paths.values(), gif]:
        assert path.exists(), path
    images = {k: Image.open(p) for k, p in paths.items()}
    width, height = images["overview"].size
    assert width == 300
    assert rs.ASPECT_RANGE[0] * 300 <= height <= rs.ASPECT_RANGE[1] * 300
    assert all(im.size == (width, height) for im in images.values())

    colors = rs.TYPE_COLORS  # (unverified grey is too close to the grey points to test)
    for kind in ("removed", "added", "moved"):
        assert near(images["overview"], colors[kind]) > 0, kind
    for kind in ("removed", "added", "moved"):  # before / after stay without boxes
        assert near(images["before"], colors[kind]) == 0, kind
        assert near(images["after"], colors[kind]) == 0, kind

    with Image.open(gif) as anim:
        assert anim.size == (width, height)
        durations = [f.info["duration"] for f in ImageSequence.Iterator(anim)]  # per seek
        assert durations == list(rs.GIF_DURATIONS_MS)
        assert anim.info.get("loop") == 0


def test_obb_convention_and_up_axis(rs):
    # three.js rotation.y: the box's local x axis points along (cos, 0, -sin)
    corners = rs.obb_corners({"obb": {"center": [1.0, 2.0, 3.0], "extents": [2.0, 0.4, 0.2],
                                      "yaw_deg": 30.0}})
    x_face = corners[4:].mean(0) - [1.0, 2.0, 3.0]  # corners 4-7: local x = +w / 2
    yaw = np.radians(30.0)
    np.testing.assert_allclose(x_face, [np.cos(yaw), 0.0, -np.sin(yaw)], atol=1e-9)
    np.testing.assert_allclose(np.ptp(corners[:, 1]), 0.4, atol=1e-9)
    # the viewer's default up axis (OpenCV, y down) becomes +Y by half a turn about z
    np.testing.assert_allclose(rs.up_rotation([0, -1, 0]), np.diag([-1.0, -1.0, 1.0]))
    tilted = np.array([0.3, 0.9, -0.2]) / np.linalg.norm([0.3, 0.9, -0.2])
    np.testing.assert_allclose(rs.up_rotation(tilted) @ tilted, [0, 1, 0], atol=1e-12)
