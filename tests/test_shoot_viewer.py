"""scripts/shoot_viewer.py on a tiny synthetic viewer folder, in headless Chromium.

Skipped without Playwright and its Chromium, or when the three.js CDN pinned in
viewer/index.html cannot be reached (the viewer loads three.js from there)."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sync_api = pytest.importorskip("playwright.sync_api")

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "shoot_viewer.py"
TIMEOUT_S = 60
CHAIR = (0.5, 0.5, 0.5)  # box sizes, metres
CHAIR_BEFORE, CHAIR_AFTER = (-1.0, 0.25, 0.0), (0.8, 0.25, -0.5)
BIN, BIN_AT = (0.3, 0.4, 0.3), (1.2, 0.2, 1.0)
PLANT, PLANT_AT = (0.3, 0.6, 0.3), (-1.0, 0.3, 1.2)
SHELF, SHELF_AT = (1.0, 1.0, 0.3), (0.0, 0.5, -1.7)


@pytest.fixture(scope="module")
def browser_ready() -> None:
    try:
        with sync_api.sync_playwright() as playwright:
            playwright.chromium.launch().close()
    except sync_api.Error as error:
        pytest.skip(f"Playwright's Chromium does not start: {error}")
    url = re.search(r'"three": "(https://[^"]+)"', (REPO / "viewer" / "index.html").read_text())[1]
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=10).close()
    except OSError as error:
        pytest.skip(f"three.js is not reachable at {url}: {error}")


def block(center: tuple, size: tuple, color: tuple, step: float = 0.05) -> tuple:
    """Points filling an axis-aligned box, all of one colour."""
    axes = [np.arange(-s / 2, s / 2 + 1e-6, step) for s in size]
    points = np.stack(np.meshgrid(*axes), -1).reshape(-1, 3) + center
    return points, np.tile(color, (len(points), 1))


def write_ply(path: Path, parts: list[tuple]) -> int:
    """ASCII PLY with float x, y, z and uchar colours; returns the number of points."""
    points = np.concatenate([p for p, _ in parts])
    colors = np.concatenate([c for _, c in parts]).astype(int)
    header = ["ply", "format ascii 1.0", f"element vertex {len(points)}",
              *(f"property float {axis}" for axis in "xyz"),
              *(f"property uchar {channel}" for channel in ("red", "green", "blue")), "end_header"]
    rows = [f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}" for (x, y, z), (r, g, b) in zip(points, colors)]
    path.write_text("\n".join(header + rows) + "\n")
    return len(points)


def box(center: tuple, size: tuple, yaw: float = 0.0) -> dict:
    lo, hi = np.subtract(center, np.divide(size, 2)), np.add(center, np.divide(size, 2))
    return {"centroid": list(center), "aabb": [lo.round(3).tolist(), hi.round(3).tolist()],
            "obb": {"center": list(center), "extents": list(size), "yaw_deg": yaw}}


def make_viewer(folder: Path) -> None:
    """Viewer app + scene.json, changes.json (one change of each type), two clouds, crops."""
    folder.mkdir(parents=True)
    for name in ("index.html", "app.js", "style.css"):
        shutil.copy2(REPO / "viewer" / name, folder / name)
    floor = np.stack(np.meshgrid(np.arange(-2, 2.01, 0.05), [0.0], np.arange(-2, 2.01, 0.05)),
                     -1).reshape(-1, 3)
    checker = ((np.floor(floor[:, 0] * 2) + np.floor(floor[:, 2] * 2)) % 2)[:, None]
    room = [(floor, np.where(checker > 0, [200, 190, 170], [120, 110, 100])),
            block((0, 0.75, -2), (4, 1.5, 0.05), (180, 180, 190))]
    blue = (40, 80, 200)
    counts = {
        "before": write_ply(folder / "before.ply", [*room, block(CHAIR_BEFORE, CHAIR, blue),
                                                    block(BIN_AT, BIN, (200, 50, 40))]),
        "after": write_ply(folder / "after.ply", [*room, block(CHAIR_AFTER, CHAIR, blue),
                                                  block(PLANT_AT, PLANT, (40, 170, 60)),
                                                  block(SHELF_AT, SHELF, (220, 180, 40))]),
    }
    (folder / "crops").mkdir()
    for side, color in (("before", (40, 80, 200)), ("after", (200, 120, 40))):
        crop = Image.new("RGB", (160, 120), color)
        ImageDraw.Draw(crop).rectangle((40, 30, 120, 90), outline=(255, 255, 255), width=3)
        crop.save(folder / "crops" / f"1_{side}.jpg")
    common = {"confidence": 0.9, "evidence": {"observing_views": 5}, "transform": None, "crops": {}}
    changes = [
        {**common, "id": 1, "type": "moved", "label": "chair",
         "before": box(CHAIR_BEFORE, CHAIR), "after": box(CHAIR_AFTER, CHAIR, 30.0),
         "transform": {"translation_m": 1.89, "rotation_deg": 30.0, "rotation_reliable": True},
         "description": "The chair moved 1.9 m.",
         "crops": {"before": "crops/1_before.jpg", "after": "crops/1_after.jpg"}},
        {**common, "id": 2, "type": "removed", "label": "bin", "before": box(BIN_AT, BIN),
         "after": None, "description": "The bin was removed."},
        {**common, "id": 3, "type": "added", "label": "plant", "before": None,
         "after": box(PLANT_AT, PLANT), "description": "A plant was added."},
        {**common, "id": 4, "type": "unverified", "label": "shelf", "before": None,
         "after": {"aabb": box(SHELF_AT, SHELF)["aabb"]}, "description": "A shelf, maybe new."},
    ]
    summary = "4 changes: 1 moved, 1 removed, 1 added; 1 unverified."
    (folder / "changes.json").write_text(json.dumps({"changes": changes, "summary": summary}))
    scene = {"name": "synthetic_room", "units": "metres", "before_ply": "before.ply",
             "after_ply": "after.ply", "changes": "changes.json",
             "stats": {"points_before": counts["before"], "points_after": counts["after"],
                       "extent_m": [4.0, 1.5, 4.0], "voxel_m": 0.02},
             "bbox_m": [[-2.0, 0.0, -2.0], [2.0, 1.5, 2.0]], "up_axis": [0.0, 1.0, 0.0],
             "view_dir": [0.0, 0.0, -1.0]}
    (folder / "scene.json").write_text(json.dumps(scene))


def run_shooter(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the script in its own process group, killed as a whole (incl. its HTTP server)
    if it takes longer than TIMEOUT_S."""
    with subprocess.Popen([sys.executable, str(SCRIPT), *args], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True) as proc:
        try:
            out, err = proc.communicate(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            pytest.fail(f"shoot_viewer.py took more than {TIMEOUT_S} s")
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def test_shoots_synthetic_viewer(tmp_path: Path, browser_ready: None) -> None:
    make_viewer(tmp_path / "run" / "viewer")
    out = tmp_path / "shots" / "syn"
    result = run_shooter(str(tmp_path / "run"), "--out", str(out), "--gif", "none",
                         "--width", "1000", "--height", "650", "--scale", "1")
    assert result.returncode == 0, result.stdout + result.stderr

    images = {}
    for name in ("viewer", "scene", "change1", "change2", "change3", "change4"):
        path = Path(f"{out}_{name}.png")
        assert path.is_file() and str(path) in result.stdout, result.stdout
        images[name] = np.asarray(Image.open(path).convert("RGB"))
        assert len(np.unique(images[name].reshape(-1, 3), axis=0)) > 1, f"{name}: one colour"
    assert images["viewer"].shape == (650, 1000, 3)
    assert not list(out.parent.glob("*.gif"))
    for name in ("change1", "change2", "change3", "change4"):  # the camera moved
        assert not np.array_equal(images[name], images["viewer"]), name

    pixels = images["scene"].reshape(-1, 3)  # the clouds cover a good part of the 3D view
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    drawn = (pixels != colors[counts.argmax()]).any(axis=1).mean()
    assert drawn > 0.05, f"only {drawn:.1%} of the 3D view differs from the background"
