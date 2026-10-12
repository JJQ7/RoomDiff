"""Tests for M3 visibility on a synthetic scene (CPU, analytic ray casting, no model).

Scene (OpenCV world, y down): a finite back wall at z = 2 m and solid 20 cm boxes in
front of it. `before` cameras stand at z = -0.5 m, `after` cameras at z = 0.3 m, both on
the same 5 x 2 grid of lateral positions and looking along +z. Depth gets Gaussian noise
so surfaces are a few voxels thick, as in real reconstructions.
"""

import json

import numpy as np
import pytest

from roomdiff.config import load_config
from roomdiff.geometry import rotation_y
from roomdiff.projection import (
    aabb_corners,
    best_views,
    project_aabb,
    project_to_pixels,
    score_views,
)
from roomdiff.recon import write_recon
from roomdiff.visibility import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    label_blobs,
    load_grid,
    make_grid,
    resolve_voxel_size,
    run_visibility,
    voxel_states,
)

H, W = 240, 320
K = np.array([[160.0, 0.0, 159.5], [0.0, 160.0, 119.5], [0.0, 0.0, 1.0]], np.float32)
WALL_Z, WALL_LO, WALL_HI = 2.0, np.array([-0.8, -0.6]), np.array([0.8, 0.6])
HALF = 0.1  # box half size (m)
BOX_A = np.array([-0.35, 0.0, 1.3])  # removed: before only
BOX_B = np.array([0.35, 0.0, 1.3])  # added: after only
BOX_C = np.array([0.45, -0.3, 0.05])  # before only, behind every after camera
GRID_XY = [(x, y) for x in (-0.8, -0.4, 0.0, 0.4, 0.8) for y in (-0.1, 0.1)]
CAM_Z = {"before": -0.5, "after": 0.3}
NOISE = 0.012  # m, depth noise

# test parameters: the defaults, with a fixed voxel size, on the CPU
PARAMS = {"voxel_size": 0.02, "device": "cpu"}


def cfg(**overrides):
    items = {**PARAMS, **overrides}
    return load_config(overrides=[f"visibility.{k}={v}" for k, v in items.items()])


def pose(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def render(cam2world: np.ndarray, boxes: list[np.ndarray], rng) -> tuple[np.ndarray, np.ndarray]:
    """z-depth and validity of the wall + solid boxes seen from an axis-aligned camera."""
    v, u = np.mgrid[0:H, 0:W]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1)
    c = cam2world[:3, 3]
    t = (WALL_Z - c[2]) / d[..., 2]
    hit = c[:2] + t[..., None] * d[..., :2]
    depth = np.where(np.all((hit >= WALL_LO) & (hit <= WALL_HI), -1), t, np.inf)
    for centre in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = (centre - HALF - c) / d
            t1 = (centre + HALF - c) / d
        near = np.minimum(t0, t1).max(-1)
        far = np.maximum(t0, t1).min(-1)
        depth = np.where((near < far) & (near > 0) & (near < depth), near, depth)
    valid = np.isfinite(depth)
    depth = np.where(valid, depth + rng.normal(0, NOISE, depth.shape), 0)
    return depth.astype(np.float32), valid


def make_scene(out, before_boxes, after_boxes, after_shift=(0.0, 0.0, 0.0), seed=0):
    """Write `<out>/recon/recon.npz` (+ a minimal meta.json) for the two box sets."""
    rng = np.random.default_rng(seed)
    data = {k: [] for k in ("rec", "files", "images", "cam2world", "K", "depth", "conf", "valid")}
    for rec_id, (name, boxes) in enumerate((("before", before_boxes), ("after", after_boxes))):
        for j, (x, y) in enumerate(GRID_XY):
            T = pose(x, y, CAM_Z[name])
            depth, valid = render(T, boxes, rng)
            stored = T.copy()
            if name == "after":
                stored[:3, 3] += after_shift  # misregistration of the whole recording
            data["rec"].append(rec_id)
            data["files"].append(f"{name}_{j:05d}.jpg")
            data["images"].append(np.where(valid[..., None], 160, 30).repeat(3, -1))
            data["cam2world"].append(stored)
            data["K"].append(K)
            data["depth"].append(depth)
            data["conf"].append(valid * 0.9)
            data["valid"].append(valid)
    recon = {k: np.stack(v) for k, v in data.items()}
    recon["metric_scale"] = 1.0
    (out / "recon").mkdir(parents=True)
    write_recon(out / "recon" / "recon.npz", recon)
    (out / "recon" / "meta.json").write_text(json.dumps({"diagnostics": {}}))
    return out


def blobs_of(out) -> list[dict]:
    return json.loads((out / "visibility" / "candidates.json").read_text())["candidates"]


def near_box(centroid, box, tol=0.06) -> bool:
    return bool(np.all(np.abs(np.asarray(centroid) - box) <= HALF + tol))


# ------------------------------------------------------------------ full stage


@pytest.fixture(scope="module")
def changed_scene(tmp_path_factory):
    out = make_scene(tmp_path_factory.mktemp("changed"), [BOX_A, BOX_C], [BOX_B])
    meta = run_visibility(out, cfg())
    return out, meta


def test_detects_removed_added_and_unobserved_boxes(changed_scene):
    out, meta = changed_scene
    blobs = blobs_of(out)
    by_type = {t: [b for b in blobs if b["type"] == t] for t in meta["blobs_per_type"]}
    assert {t: len(b) for t, b in by_type.items()} == {
        "disappeared": 1, "appeared": 1, "unverified_before": 1, "unverified_after": 0}
    removed, added, unseen = by_type["disappeared"][0], by_type["appeared"][0], \
        by_type["unverified_before"][0]
    assert near_box(removed["centroid"], BOX_A) and near_box(added["centroid"], BOX_B)
    assert near_box(unseen["centroid"], BOX_C)
    # the other recording saw through the changed boxes, but never looked at box C
    assert removed["evidence"]["observing_views"] >= 2 and added["evidence"]["observing_views"] >= 2
    assert removed["evidence"]["free_frames_mean"] >= 2
    assert unseen["evidence"]["observing_views"] == 0
    assert unseen["evidence"]["free_frames_mean"] < 2
    assert all(b["evidence"]["occ_points"] > 0 for b in blobs)
    assert [b["id"] for b in blobs] == [1, 2, 3]
    assert [b["volume_m3"] for b in blobs] == sorted((b["volume_m3"] for b in blobs), reverse=True)


def test_outputs_match_the_contract(changed_scene):
    out, meta = changed_scene
    grid = load_grid(out / "visibility" / "grid.npz")
    dims = tuple(grid["dims"])
    assert grid["voxel_size"] == pytest.approx(0.02) and grid["origin"].shape == (3,)
    for name in ("before", "after"):
        assert grid[f"state_{name}"].shape == dims and grid[f"state_{name}"].dtype == np.uint8
        assert grid[f"free_count_{name}"].dtype == np.int16
        assert grid[f"occ_count_{name}"].dtype == np.int32
        states = meta["voxels_per_state"][name]
        assert sum(states.values()) == np.prod(dims) and states["free"] > 0
    for blob in blobs_of(out):
        labels = grid[f"labels_{blob['type']}"]
        assert labels.dtype == np.int32 and (labels == blob["id"]).sum() == blob["voxels"]
        lo, hi = np.array(blob["aabb"])
        assert np.all(lo <= blob["centroid"]) and np.all(np.array(blob["centroid"]) <= hi)
        crops = sorted((out / "visibility" / "debug").glob(f"blob_{blob['id']}_*.jpg"))
        views = blob["views"]
        assert len(crops) == len(views["before"]) + len(views["after"])
        assert all((out / "visibility" / v["crop"]).exists() for v in views["before"])
        # 2 views where the blob is occupied; none of the after cameras sees box C
        assert len(views[{"disappeared": "before", "appeared": "after"}.get(
            blob["type"], "before")]) == 2
        if blob["type"] == "unverified_before":
            assert views["after"] == []
    assert meta["peak_vram_gb"] is None and meta["device"] == "cpu"
    assert meta["params"]["voxel_size"] == 0.02 and meta["voxel_size_source"] == "fixed"
    assert json.loads((out / "visibility" / "meta.json").read_text()) == meta


def test_no_blobs_without_change(tmp_path):
    out = make_scene(tmp_path, [BOX_A, BOX_B], [BOX_A, BOX_B])
    meta = run_visibility(out, cfg())
    assert blobs_of(out) == [] and sum(meta["blobs_per_type"].values()) == 0


def test_distance_margin_suppresses_a_3cm_shift(tmp_path):
    out = make_scene(tmp_path, [BOX_A, BOX_B], [BOX_A, BOX_B], after_shift=(0.03, 0.0, 0.03))
    meta = run_visibility(out, cfg())
    assert sum(meta["blobs_per_type"].values()) == 0
    # without the margin (and without the opening) the shift does produce candidates
    meta = run_visibility(out, cfg(dist_margin=0.0, opening_size=1))
    assert sum(meta["blobs_per_type"].values()) > 0
    assert meta["candidate_voxels"]["disappeared"]["raw"] > 0


def test_cli_visibility_and_viz(tmp_path):
    from typer.testing import CliRunner

    from roomdiff.cli import app

    out = make_scene(tmp_path, [BOX_A], [BOX_B])
    runner = CliRunner()
    result = runner.invoke(app, ["visibility", "--out", str(out), "-s", "visibility.device=cpu",
                                 "-s", "visibility.voxel_size=0.02"])
    assert result.exit_code == 0, result.output
    assert "disappeared 1, appeared 1" in result.output
    assert (out / "visibility" / "visibility.rrd").stat().st_size > 1000
    result = runner.invoke(app, ["viz", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "2 candidate blobs" in result.output and (out / "scene.rrd").exists()


# ------------------------------------------------------------------ helpers


def test_resolve_voxel_size():
    meta = {"diagnostics": {"nn_covisible": {"before_to_after": {"median_m": 0.015},
                                             "after_to_before": {"median_m": 0.012}}}}
    assert resolve_voxel_size("auto", meta)[0] == pytest.approx(0.03)
    assert resolve_voxel_size("auto", {"diagnostics": {"nn_covisible": {
        "before_to_after": {"median_m": 0.004}}}})[0] == pytest.approx(0.02)
    assert resolve_voxel_size("auto", {})[0] == pytest.approx(0.02)
    assert resolve_voxel_size(0.04, meta) == (0.04, "fixed")
    with pytest.raises(ValueError, match="number or 'auto'"):
        resolve_voxel_size("big", meta)


def test_voxel_size_override_replaces_auto():
    assert load_config().visibility.voxel_size == "auto"
    assert load_config(overrides=["visibility.voxel_size=0.03"]).visibility.voxel_size == 0.03


def test_make_grid_respects_max_voxels():
    grid, grew = make_grid(np.zeros(3), np.ones(3), 0.02, 10**9, pad=3)
    assert not grew and grid.dims == (56, 56, 56) and np.allclose(grid.origin, -0.06)
    grid, grew = make_grid(np.zeros(3), np.ones(3), 0.02, 50**3, pad=3)
    assert grew and grid.size <= 50**3 and grid.voxel_size > 0.02
    ijk, inside = grid.voxel_of(np.array([[0.5, 0.5, 0.5], [5.0, 0.0, 0.0]]))
    assert inside.tolist() == [True, False]
    assert np.all(np.abs(grid.centers(ijk[:1]) - 0.5) <= grid.voxel_size / 2)


def test_voxel_states_priority():
    occ = np.array([0, 5, 0, 1])
    free = np.array([0, 9, 3, 1])
    assert voxel_states(occ, free, 3, 2).tolist() == [UNKNOWN, OCCUPIED, FREE, UNKNOWN]


def test_label_blobs_opening_and_size_filter():
    mask = np.zeros((20, 20, 20), bool)
    mask[2:7, 2:7, 2:7] = True  # solid 5^3 cube: survives the opening
    mask[10:18, 10, 10:18] = True  # one-voxel-thick sheet: removed by the opening
    mask[15, 2, 2] = True  # single voxel
    labels, kept, left = label_blobs(mask, 0.02, 3, 0.0, 8)
    assert len(kept) == 1 and left == 125 and (labels == kept[0]).sum() == 125
    labels, kept, _ = label_blobs(mask, 0.02, 0, 0.0, 8)  # no opening: cube + sheet
    assert sorted((labels == k).sum() for k in kept) == [64, 125]
    _, kept, _ = label_blobs(mask, 0.02, 0, 0.0015, 8)  # 125 voxels = 0.001 m^3 < min
    assert kept == []
    shell = np.zeros((20, 20, 20), bool)
    shell[2:12, 2:12, 5:7] = True  # two-voxel-thick shell: kept by the default 2^3 opening
    shell[15, 15, 15] = True
    labels, kept, left = label_blobs(shell, 0.02, 2, 0.0, 8)
    assert left == 200 and len(kept) == 1 and labels[2:12, 2:12, 5:7].all()
    assert label_blobs(shell, 0.02, 3, 0.0, 8)[1] == []  # a 3^3 opening erases it


def test_projection_helpers():
    T = pose(0.0, 0.0, 0.0)
    col, row, z, inside = project_to_pixels(np.array([[0, 0, 2.0], [0, 0, -1.0]]), K, T, W, H)
    assert inside.tolist() == [True, False] and (col[0], row[0]) == (160, 120)
    assert z.tolist() == [2.0, -1.0]
    assert aabb_corners([0, 0, 0], [1, 2, 3]).max(0).tolist() == [1, 2, 3]
    box = project_aabb([-0.1, -0.1, 1.9], [0.1, 0.1, 2.1], K, T, W, H)
    assert box is not None and box[0] < 159.5 < box[2] and box[1] < 119.5 < box[3]
    assert project_aabb([-1, -1, -3], [1, 1, -2], K, T, W, H) is None  # behind the camera
    # a 20 cm patch on a wall 2 m away: camera 1 is farther back (smaller), camera 2 sees an
    # occluder 1 m in front of it, camera 3 stands past it, camera 4 looks the other way
    patch = np.stack(np.meshgrid(np.linspace(-0.1, 0.1, 21), np.linspace(-0.1, 0.1, 21),
                                 [2.0], indexing="ij"), -1).reshape(-1, 3)
    flip = pose(0, 0, 0)
    flip[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    cams = np.stack([pose(0, 0, 0), pose(0.3, 0, -1.0), pose(0.2, 0, 0), pose(0, 0, 3.0), flip])
    depth = np.full((5, H, W), 2.0, np.float32)
    depth[1] = 3.0
    depth[2] = 1.0
    views = best_views(patch, range(5), cams, np.repeat(K[None], 5, 0), depth, k=4)
    assert [v.frame for v in views] == [0, 1, 2]  # cameras 3 and 4 never see the patch
    assert views[0].visible == 1.0 and not views[0].occluded and not views[0].cut
    assert views[0].score > views[1].score > 0 and views[2].occluded and views[2].score == 0
    assert views[0].box == (152, 112, 168, 128)  # all projected points, also when occluded
    assert views[2].box == (136, 112, 152, 128)
    # the place seen by the other recording: visible = not occluded (seen past)
    behind = np.where(depth == 2.0, 3.0, depth)  # camera 0 sees the wall behind the place
    place = score_views(patch, [0, 2], cams, np.repeat(K[None], 5, 0), behind,
                        on_surface=False)
    assert [(v.frame, v.visible) for v in place] == [(0, 1.0), (2, 0.0)]
    assert best_views(patch[:0], range(5), cams, np.repeat(K[None], 5, 0), depth) == []


def look_at(centre, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """OpenCV camera (x right, y down, z forward) at `centre` looking at `target` in a
    y-up world."""
    f = np.asarray(target, float) - centre
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(f, x), f], 1)
    T[:3, 3] = centre
    return T


def splat_depth(points, cam2world) -> np.ndarray:
    """z-buffer of `points` in front of a background 6 m away."""
    depth = np.full((H, W), 6.0, np.float32)
    col, row, z, inside = project_to_pixels(points, K, cam2world, W, H)
    order = np.argsort(-z[inside])  # far first: the nearest point is written last
    depth[row[inside][order], col[inside][order]] = z[inside][order]
    return depth


def test_view_ranking_does_not_depend_on_the_world_frame():
    """Rotating points and cameras about +Y together leaves the depth maps (camera frame)
    unchanged, and must leave the ranking unchanged too (it used to rank by the projected
    axis-aligned box, which depends on the world axes)."""
    rng = np.random.default_rng(3)
    # a bench 0.9 x 0.4 x 0.3 m (surface samples), turned 30 degrees, at the origin
    u = np.linspace(-0.5, 0.5, 46)
    faces = [np.stack([a, b, c], -1).reshape(-1, 3) for a, b, c in (
        np.meshgrid(0.9 * u, 0.4 * u, [-0.15, 0.15], indexing="ij"),
        np.meshgrid(0.9 * u, [-0.2, 0.2], 0.3 * u, indexing="ij"),
        np.meshgrid([-0.45, 0.45], 0.4 * u, 0.3 * u, indexing="ij"))]
    bench = np.concatenate(faces) @ rotation_y(30.0).T + [0.0, 0.2, 0.0]
    cams = []
    for k in range(12):  # all around, 0.6-3 m away, aimed a little off the bench
        a = 2 * np.pi * k / 12 + rng.uniform(-0.2, 0.2)
        c = np.array([np.sin(a), 0, np.cos(a)]) * rng.uniform(0.6, 3.0) + [0, 0.8, 0]
        cams.append(look_at(c, rng.uniform(-0.8, 0.8, 3) + [0, 0.2, 0]))
    cams = np.stack(cams)
    depth = np.stack([splat_depth(bench, T) for T in cams])
    Ks = np.repeat(K[None], len(cams), 0)
    frames = range(len(cams))
    ref = score_views(bench, frames, cams, Ks, depth)
    assert len(ref) >= 8 and ref[0].score > 0
    assert any(v.cut for v in ref) and not all(v.cut for v in ref)

    def aabb_order(points, poses):
        lo, hi = points.min(0), points.max(0)
        boxes = [project_aabb(lo, hi, K, T, W, H) for T in poses]
        area = [(b[2] - b[0]) * (b[3] - b[1]) if b else 0 for b in boxes]
        return list(np.argsort(area, kind="stable"))

    old_rank_changed = False
    for yaw in rng.uniform(-180, 180, 4):
        turn = np.eye(4)
        turn[:3, :3] = rotation_y(yaw)
        turned = score_views(bench @ turn[:3, :3].T, frames, turn @ cams, Ks, depth)
        assert [v.frame for v in turned] == [v.frame for v in ref]
        assert [v.box for v in turned] == [v.box for v in ref]
        assert np.allclose([v.score for v in turned], [v.score for v in ref], rtol=1e-6)
        old_rank_changed |= aabb_order(bench @ turn[:3, :3].T, turn @ cams) != \
            aabb_order(bench, cams)
    assert old_rank_changed  # the axis-aligned box ranking does depend on the frame
