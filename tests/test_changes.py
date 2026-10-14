"""Tests for M4 matching, change typing and changes.json (CPU, fake models).

End-to-end scene (OpenCV world, y down): a dark back wall at z = 2 m and coloured 20 cm
boxes. Before: red at R0, blue at B; after: red at R1 (moved 0.8 m), green at G (added).
A yellow box Y stands behind every after camera (seen only before: unverified). Cameras
as in tests/test_visibility.py: before at z = -0.5 m, after at z = 0.3 m.
"""

import json

import numpy as np
import pytest

from roomdiff.changes import (
    box_extents,
    confidence,
    describe,
    extents_compatible,
    match_instances,
    rigid_transform,
    summarize,
)
from roomdiff.config import load_config
from roomdiff.instances import decode_rle, load_instances, load_masks, run_instances
from roomdiff.models import FakeFeaturizer, FakeSegmenter
from roomdiff.recon import write_recon
from roomdiff.visibility import run_visibility

H, W = 240, 320
K = np.array([[160.0, 0.0, 159.5], [0.0, 160.0, 119.5], [0.0, 0.0, 1.0]], np.float32)
WALL_Z, WALL_LO, WALL_HI = 2.0, np.array([-0.9, -0.7]), np.array([0.9, 0.7])
HALF = 0.1
COLORS = {0: (20, 20, 20), 1: (220, 30, 30), 2: (30, 30, 220), 3: (30, 200, 30),
          4: (220, 220, 30)}
R0, R1 = np.array([-0.4, -0.2, 1.3]), np.array([0.4, -0.2, 1.3])
B, G, Y = np.array([-0.4, 0.25, 1.4]), np.array([0.4, 0.25, 1.4]), np.array([0.45, -0.3, 0.05])
SCENE = {"before": [(R0, 1), (B, 2), (Y, 4)], "after": [(R1, 1), (G, 3)]}
GRID_XY = [(x, y) for x in (-0.8, -0.4, 0.0, 0.4, 0.8) for y in (-0.1, 0.1)]
CAM_Z = {"before": -0.5, "after": 0.3}
SETTINGS = ["visibility.device=cpu", "visibility.voxel_size=0.02", "instances.device=cpu",
            "instances.min_unverified_volume=0.001"]

REQUIRED = {"id": int, "type": str, "label": str, "confidence": float, "evidence": dict,
            "description": str, "crops": dict}
SIDE_KEYS = {"instance_id": int, "centroid": list, "aabb": list, "extents": list, "obb": dict}
TRANSFORM_KEYS = {"translation_m": float, "R": list, "t": list}


def render(c: np.ndarray, boxes, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z-depth, validity and object ids (0 = wall, -1 = nothing) seen from camera centre c
    looking along +z."""
    v, u = np.mgrid[0:H, 0:W]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1)
    t = (WALL_Z - c[2]) / d[..., 2]
    hit = c[:2] + t[..., None] * d[..., :2]
    on_wall = np.all((hit >= WALL_LO) & (hit <= WALL_HI), -1)
    depth = np.where(on_wall, t, np.inf)
    ids = np.where(on_wall, 0, -1)
    for centre, obj in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t0, t1 = (centre - HALF - c) / d, (centre + HALF - c) / d
        near, far = np.minimum(t0, t1).max(-1), np.maximum(t0, t1).min(-1)
        hit_box = (near < far) & (near > 0) & (near < depth)
        depth = np.where(hit_box, near, depth)
        ids = np.where(hit_box, obj, ids)
    valid = np.isfinite(depth)
    depth = np.where(valid, depth + rng.normal(0, 0.008, depth.shape), 0)
    return depth.astype(np.float32), valid, ids


def make_scene(out) -> dict[str, np.ndarray]:
    """Write `<out>/recon/` for SCENE; returns the label maps keyed by frame index."""
    rng = np.random.default_rng(0)
    data = {k: [] for k in ("rec", "files", "images", "cam2world", "K", "depth", "conf", "valid")}
    labels = {}
    for rec_id, name in enumerate(("before", "after")):
        for j, (x, y) in enumerate(GRID_XY):
            T = np.eye(4)
            T[:3, 3] = [x, y, CAM_Z[name]]
            depth, valid, ids = render(T[:3, 3], SCENE[name], rng)
            image = np.zeros((H, W, 3), np.uint8)
            for obj, color in COLORS.items():
                image[ids == obj] = color
            labels[str(len(data["rec"]))] = ids
            data["rec"].append(rec_id)
            data["files"].append(f"{name}_{j:05d}.jpg")
            data["images"].append(image)
            data["cam2world"].append(T)
            data["K"].append(K)
            data["depth"].append(depth)
            data["conf"].append(valid * 0.9)
            data["valid"].append(valid)
    recon = {k: np.stack(v) for k, v in data.items()}
    recon["metric_scale"] = 1.0
    (out / "recon").mkdir(parents=True)
    write_recon(out / "recon" / "recon.npz", recon)
    (out / "recon" / "meta.json").write_text(json.dumps({"diagnostics": {}}))
    return labels


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    from roomdiff.changes import run_changes

    out = tmp_path_factory.mktemp("changes")
    labels = make_scene(out)
    cfg = load_config(overrides=SETTINGS)
    run_visibility(out, cfg)
    meta = run_instances(out, cfg, FakeSegmenter(labels), FakeFeaturizer())
    doc = run_changes(out, cfg)
    return out, meta, doc


def change_of(doc, kind):
    return [c for c in doc["changes"] if c["type"] == kind]


# ------------------------------------------------------------------ end to end


def test_scene_changes_have_the_right_types(scene):
    _, meta, doc = scene
    assert meta["peak_vram_gb"] is None
    moved, removed, added = (change_of(doc, k) for k in ("moved", "removed", "added"))
    assert len(moved) == 1 and len(removed) == 1 and len(added) == 1
    assert np.allclose(moved[0]["before"]["centroid"], R0, atol=0.12)
    assert np.allclose(moved[0]["after"]["centroid"], R1, atol=0.12)
    assert moved[0]["transform"]["translation_m"] == pytest.approx(0.8, abs=0.08)
    rot = moved[0]["transform"]["rotation_deg"]
    assert rot is None or rot < 10
    assert moved[0]["evidence"]["appearance_similarity"] > 0.9
    assert np.allclose(removed[0]["before"]["centroid"], B, atol=0.12)
    assert removed[0]["after"] is None and removed[0]["evidence"]["observing_views"] >= 2
    assert np.allclose(added[0]["after"]["centroid"], G, atol=0.12)
    unverified = change_of(doc, "unverified")
    seen = [c for c in unverified if np.allclose(c["before"]["centroid"], Y, atol=0.12)]
    assert len(seen) == 1 and seen[0]["evidence"]["seen_in"] == "before"
    assert seen[0]["confidence"] <= 0.5
    assert all(c["confidence"] <= 0.5 for c in unverified)
    assert doc["summary"].startswith("3 changes detected: 1 moved, 1 added, 1 removed")


def test_changes_json_schema(scene):
    out, _, doc = scene
    assert json.loads((out / "changes.json").read_text()) == doc
    assert set(doc) == {"scene", "changes", "summary"} and doc["scene"]["units"]
    assert [c["id"] for c in doc["changes"]] == list(range(1, len(doc["changes"]) + 1))
    for c in doc["changes"]:
        for key, kind in REQUIRED.items():
            assert isinstance(c[key], kind), (key, c[key])
        assert c["type"] in ("moved", "added", "removed", "unverified")
        assert 0.0 <= c["confidence"] <= 1.0 and c["label"] == "object"
        assert "before" in c and "after" in c and "transform" in c
        sides = [s for s in ("before", "after") if c[s] is not None]
        assert sides == {"moved": ["before", "after"], "removed": ["before"],
                         "added": ["after"]}.get(c["type"], sides) and len(sides) >= 1
        for s in sides:
            for key, kind in SIDE_KEYS.items():
                assert isinstance(c[s][key], kind)
            lo, hi = np.array(c[s]["aabb"])
            assert lo.shape == (3,) and np.all(lo <= c[s]["centroid"])
            assert np.all(np.array(c[s]["centroid"]) <= hi)
            obb = c[s]["obb"]  # vertical box: same height as the AABB, footprint no larger
            assert set(obb) == {"center", "extents", "yaw_deg"} and -45 < obb["yaw_deg"] <= 45
            assert obb["extents"][1] == pytest.approx(hi[1] - lo[1], abs=1e-3)
            w, _, d = obb["extents"]
            assert w * d <= (hi[0] - lo[0]) * (hi[2] - lo[2]) + 1e-6
        if c["type"] == "moved":
            for key, kind in TRANSFORM_KEYS.items():
                assert isinstance(c["transform"][key], kind)
            assert np.array(c["transform"]["R"]).shape == (3, 3)
        else:
            assert c["transform"] is None
        for key in ("free_rays", "observing_views", "vlm_verdict"):
            assert key in c["evidence"]
        for side, path in c["crops"].items():
            assert side in ("before", "after", "before_place", "after_place")
            assert path is None or (out / path).is_file()
        assert c["crops"].get("before") or c["crops"].get("after")
    moved = change_of(doc, "moved")[0]
    assert set(moved["crops"]) == {"before", "after", "before_place", "after_place"}


def test_instance_files(scene):
    out, meta, _ = scene
    assert meta["instances"]["before"] >= 3 and meta["instances"]["after"] >= 2
    for name in ("before", "after"):
        data = load_instances(out, name)
        assert data["feats"].shape == (len(data["instances"]), FakeFeaturizer.dim)
        assert np.allclose(np.linalg.norm(data["feats"], axis=1), 1.0, atol=1e-5)
        masks = load_masks(out / "instances" / f"{name}_masks.npz")
        assert masks, "per-keyframe masks expected"
        for record in data["instances"]:
            assert record["n_points"] == len(data["points"][record["id"]]) >= 150
            assert record["feature_index"] == record["id"] - 1
            assert record["views"] and all(v["file"].startswith(name) for v in record["views"])
        entries = next(iter(masks.values()))
        inst_id, score, rle = entries[0]
        mask = decode_rle(rle)
        assert mask.shape == (H, W) and mask.any() and 0 < score <= 1
        assert inst_id in data["points"]


def test_unverified_changes_say_why(scene):
    _, _, doc = scene
    unverified = change_of(doc, "unverified")
    assert unverified and all(c["evidence"]["reason_class"] == "never_observed"
                              for c in unverified)
    assert all("never observed" in c["description"] for c in unverified)
    assert not any("reason_class" in c["evidence"] for c in doc["changes"]
                   if c["type"] != "unverified")


def test_weak_alignment_turns_geometry_changes_unverified(tmp_path):
    """With a WEAK recon verdict M3 demotes the disappeared / appeared blobs: the moved red
    box, the removed blue one and the added green one come back as unverified regions with
    `reason_class` "weak_alignment"; the yellow box the after video never saw stays
    "never_observed"."""
    from roomdiff.changes import run_changes

    labels = make_scene(tmp_path)
    (tmp_path / "recon" / "meta.json").write_text(json.dumps(
        {"diagnostics": {"verdict": "alignment WEAK", "verdict_reason": "test"}}))
    cfg = load_config(overrides=[*SETTINGS, "visibility.weak_alignment_demotes=true"])
    run_visibility(tmp_path, cfg)
    run_instances(tmp_path, cfg, FakeSegmenter(labels), FakeFeaturizer())
    doc = run_changes(tmp_path, cfg)
    assert {c["type"] for c in doc["changes"]} == {"unverified"}
    assert doc["summary"].startswith("0 changes detected: 0 moved, 0 added, 0 removed")

    def at(centre, side):
        found = [c for c in doc["changes"] if c[side] is not None
                 and np.allclose(c[side]["centroid"], centre, atol=0.12)]
        assert len(found) == 1, (centre, side)
        return found[0]

    for centre, side in ((R0, "before"), (B, "before"), (R1, "after"), (G, "after")):
        change = at(centre, side)
        assert change["evidence"]["seen_in"] == side and change["confidence"] <= 0.5
        assert change["evidence"]["reason_class"] == "weak_alignment"
        assert "too poorly aligned" in change["description"]
    assert at(Y, "before")["evidence"]["reason_class"] == "never_observed"
    blobs = json.loads((tmp_path / "visibility" / "candidates.json").read_text())["candidates"]
    demoted = {b["id"] for b in blobs if b.get("demotion_reason") == "weak_alignment"}
    assert len(demoted) >= 4
    for name in ("before", "after"):
        for record in load_instances(tmp_path, name)["instances"]:
            assert record["kind"] == "unverified"
            assert record["demoted"] == ("weak_alignment" if demoted & set(record["blob_ids"])
                                         else None)


def test_cli_instances_changes_viz(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import roomdiff.models
    from roomdiff.cli import app

    labels = make_scene(tmp_path)
    monkeypatch.setattr(roomdiff.models, "load_models",
                        lambda params: (FakeSegmenter(labels), FakeFeaturizer()))
    runner = CliRunner()
    sets = [x for s in SETTINGS for x in ("-s", s)]
    for command in ("visibility", "instances", "changes"):
        result = runner.invoke(app, [command, "--out", str(tmp_path), *sets])
        assert result.exit_code == 0, result.output
    assert "1 moved, 1 added, 1 removed" in result.output
    assert (tmp_path / "changes.rrd").stat().st_size > 1000
    result = runner.invoke(app, ["viz", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "changes)" in result.output and (tmp_path / "scene.rrd").exists()


# ------------------------------------------------------------------ units


PARAMS = dict(load_config().changes)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_hungarian_matching_with_thresholds():
    feats_d = np.stack([unit([1, 0, 0, 0]), unit([0, 1, 0, 0]), unit([0, 0, 1, 0])])
    feats_a = np.stack([unit([0.1, 1, 0, 0]), unit([1, 0.05, 0, 0]), unit([0, 0, 0.6, 1])])
    ext = [[0.4, 0.5, 0.6]] * 3
    pairs, cos, cost = match_instances(feats_d, feats_a, ext, ext, PARAMS)
    # d0 <-> a1 and d1 <-> a0 match (moved); d2 / a2 are too dissimilar: removed / added
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (1, 0)]
    assert cos.shape == (3, 3) and cost[2, 2] >= 1e6
    # a size mismatch (sorted extents ratio > 2) blocks an otherwise perfect match
    pairs, _, _ = match_instances(feats_d[:1], feats_d[:1], [[0.2, 0.2, 0.2]],
                                  [[0.2, 0.2, 0.5]], PARAMS)
    assert pairs == []
    # extents are compared sorted: a rotated object still matches
    assert extents_compatible([0.2, 0.5, 0.9], [0.9, 0.21, 0.48], 0.5, 2.0)
    # the Hungarian assignment prefers the globally cheaper pairing
    feats_a2 = np.stack([unit([1, 0.9, 0, 0]), unit([1, 0, 0, 0])])
    pairs, _, _ = match_instances(feats_d[:2], feats_a2, ext[:2], ext[:2], PARAMS)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (1, 0)]
    assert match_instances(feats_d[:0], feats_a, [], ext, PARAMS)[0] == []
    # sizes come from the oriented box (frame-independent), not the axis-aligned extents
    record = {"extents": [0.7, 0.5, 0.7], "obb": {"extents": [0.5, 0.5, 0.4]}}
    assert box_extents(record) == [0.5, 0.5, 0.4]
    assert box_extents({"extents": [0.7, 0.5, 0.7], "obb": None}) == [0.7, 0.5, 0.7]


def test_confidence_bounds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 3))
        masks = rng.uniform(0, 1.2, n).tolist()
        cons = [None if rng.random() < 0.3 else float(rng.uniform(-0.2, 1.0)) for _ in range(n)]
        views = rng.integers(0, 40, n).tolist()
        c = confidence(masks, cons, views, PARAMS)
        assert 0.0 <= c <= 1.0
        assert 0.0 <= confidence(masks, cons, None, PARAMS) <= 0.5
    assert confidence([1.0], [1.0], [10], PARAMS) == pytest.approx(1.0)
    assert confidence([1.0], [1.0], None, PARAMS) == pytest.approx(0.5)
    assert confidence([0.9], [0.8], [1], PARAMS) < confidence([0.9], [0.8], [5], PARAMS)
    assert confidence([0.9], [0.8], [5], PARAMS) == confidence([0.9], [0.8], [50], PARAMS)


LEGACY = {**PARAMS, "w_mask": 0.3, "w_free": 0.4, "w_consistency": 0.3,
          "appearance_evidence": "frames"}  # the confidence before 2026-10-14


def test_confidence_weights_default_and_legacy():
    """Default: 0.6 mask score + 0.4 free-space evidence, view consistency unused; the
    earlier weights (0.3 / 0.4 / 0.3) stay selectable by configuration."""
    assert confidence([0.9], [0.2], [3], PARAMS) == pytest.approx(0.6 * 0.9 + 0.4 * 0.6)
    assert confidence([0.9], [0.2], [3], PARAMS) == confidence([0.9], [0.95], [3], PARAMS)
    assert confidence([0.9], [0.8], [3], LEGACY) == pytest.approx(0.3 * 0.9 + 0.4 * 0.6 + 0.3 * 0.8)
    assert confidence([0.9], [0.8], None, PARAMS) == pytest.approx(0.45)  # unverified: 0.5 S_mask
    assert confidence([0.9], [0.8], None, LEGACY) == pytest.approx(0.425)
    # moved: mean mask score and mean free-space term of both places
    assert confidence([0.8, 1.0], [None, 0.5], [2, 10], PARAMS) == pytest.approx(0.6 * 0.9
                                                                               + 0.4 * 0.7)


def test_change_confidence_is_monotone_and_bounded():
    """confidence x appearance factor, as run_changes combines them: in [0, 1] (unverified
    <= 0.5) and non-decreasing in the mask score, the observing views and the cue."""
    from roomdiff.changes import appearance_support

    rng = np.random.default_rng(3)
    for params in (PARAMS, LEGACY):
        for _ in range(100):
            mask, cons = float(rng.uniform(0, 1)), float(rng.uniform(-0.2, 1))
            views, cue = int(rng.integers(0, 30)), float(rng.uniform(0, 2.5))
            frames = int(rng.integers(0, 10))

            def conf(m=mask, v=views, c=cue, f=frames, k=cons, p=params):
                record = {"evidence": {"appearance": {"mean_score": c, "frames_supporting": f}}}
                return confidence([m], [k], [v], p) * appearance_support(record, p)

            assert 0.0 <= conf() <= 1.0
            assert confidence([mask], [cons], None, params) <= 0.5
            assert conf(m=min(mask + 0.1, 1.0)) >= conf()
            assert conf(v=views + 1) >= conf()
            assert conf(c=cue + 0.1) >= conf() and conf(f=frames + 1) >= conf()


def test_rigid_transform_recovers_rotation_and_translation():
    rng = np.random.default_rng(1)
    # an L-shaped chair-like point set (seat + back), 1 cm noise-free grid
    seat = np.stack(np.meshgrid(np.arange(0, 0.4, 0.01), [0.45], np.arange(0, 0.4, 0.01),
                                indexing="ij"), -1).reshape(-1, 3)
    back = np.stack(np.meshgrid(np.arange(0, 0.4, 0.01), np.arange(0, 0.45, 0.01), [0.4],
                                indexing="ij"), -1).reshape(-1, 3)
    src = np.concatenate([seat, back]) + rng.normal(0, 0.002, (len(seat) + len(back), 3))
    angle = np.radians(25)
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    c = src.mean(0)
    dst = (src - c) @ R.T + c + [0.6, 0.0, 0.3]
    tf = rigid_transform(src, dst, 0.10, 0.3, 0.5)
    assert tf["rotation_deg"] == pytest.approx(25, abs=2) and tf["rotation_reliable"]
    assert tf["icp_fitness"] > 0.9
    assert tf["translation_m"] == pytest.approx(np.hypot(0.6, 0.3), abs=0.02)
    assert np.allclose(np.array(tf["R"]) @ np.array(tf["R"]).T, np.eye(3), atol=1e-5)
    # unrelated shapes: no trustworthy fit, rotation unknown, centroid translation
    blob = rng.normal(0, 0.15, (500, 3)) + [3, 0, 0]
    tf = rigid_transform(src, blob, 0.10, 0.9)
    assert tf["rotation_deg"] is None and not tf["rotation_reliable"]
    assert tf["translation_m"] == pytest.approx(np.linalg.norm(blob.mean(0) - src.mean(0)),
                                                abs=1e-3)


def test_rotation_between_the_fitness_thresholds_is_reported_as_unreliable():
    """Fitness >= icp_reliable_fitness: reliable angle; icp_min_fitness <= fitness < it: the
    angle is reported but marked unreliable (and not quoted in the description)."""
    rng = np.random.default_rng(2)
    src = rng.uniform([0, 0, 0], [0.4, 1.4, 0.1], (3000, 3))
    angle = np.radians(15)
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    dst = (src - src.mean(0)) @ R.T + src.mean(0) + [0.3, 0.0, 0.1]
    fit = rigid_transform(src, dst, 0.10, 0.3, 0.5)["icp_fitness"]
    shaky = rigid_transform(src, dst, 0.10, fit - 0.01, fit + 0.01)  # just below "reliable"
    assert shaky["rotation_deg"] == pytest.approx(15, abs=2) and not shaky["rotation_reliable"]
    assert not np.allclose(shaky["R"], np.eye(3))  # the ICP motion is used
    change = {"type": "moved", "evidence": {}, "transform": shaky,
              "before": {"extents": [0.4, 1.4, 0.1], "centroid": [0, 0, 0]},
              "after": {"extents": [0.4, 1.4, 0.1], "centroid": [0.3, 0, 0.1]}}
    assert describe("moved", change).endswith("(rotation uncertain).")
    change["transform"] = {**shaky, "rotation_reliable": True}
    assert "turned by about 15 degrees" in describe("moved", change)


def test_summary_text():
    changes = [{"type": t} for t in ("moved", "added", "added", "unverified")]
    assert summarize(changes) == "3 changes detected: 1 moved, 2 added, 0 removed; " \
        "1 region unverified."
    assert summarize([]).endswith("0 regions unverified.")


def test_rerun_boxes_use_the_oriented_box():
    from roomdiff.viz_rerun import box_args, box_centre

    side = {"centroid": [0.0, 0.0, 0.0], "aabb": [[-1.0, 0.0, -1.0], [1.0, 1.0, 1.0]],
            "obb": {"center": [0.1, 0.5, 0.0], "extents": [2.0, 1.0, 0.4], "yaw_deg": 30.0}}
    args = box_args(side)
    assert args["centers"] == [[0.1, 0.5, 0.0]] and np.allclose(args["half_sizes"], [[1, 0.5, 0.2]])
    assert len(args["rotation_axis_angles"]) == 1
    assert np.allclose(box_centre(side), [0.1, 0.5, 0.0])
    del side["obb"]
    args = box_args(side)
    assert np.allclose(args["centers"], [[0, 0.5, 0]]) and "rotation_axis_angles" not in args
    assert np.allclose(box_centre(side), [0, 0, 0])


def test_appearance_support_factor():
    """1 for a geometry instance; an appearance instance: 0.5 + 0.5 min(cue / full, 1) with
    the cluster's mean region score (default, full 2) or, as before, frames_supporting."""
    from roomdiff.changes import appearance_support

    score = {"appearance_evidence": "score", "appearance_score_full": 2.0}
    frames = {"appearance_evidence": "frames", "appearance_frames_full": 4}
    for params in (score, frames, PARAMS):
        assert appearance_support({"evidence": {}}, params) == 1.0
        assert appearance_support({"evidence": {"appearance": None}}, params) == 1.0

    def app(**terms):
        return {"evidence": {"appearance": {"frames_supporting": 2, "mean_score": 1.2, **terms}}}

    assert appearance_support(app(), frames) == 0.75
    assert appearance_support(app(frames_supporting=8), frames) == 1.0
    assert appearance_support(app(), score) == pytest.approx(0.8)
    assert appearance_support(app(), PARAMS) == pytest.approx(0.8)  # the default
    assert appearance_support(app(mean_score=0.0), score) == 0.5
    assert appearance_support(app(mean_score=2.6), score) == 1.0
    values = [appearance_support(app(mean_score=float(c)), score) for c in np.linspace(0, 3, 31)]
    assert all(0.5 <= v <= 1.0 for v in values) and np.all(np.diff(values) >= 0)
    with pytest.raises(ValueError, match="appearance_evidence"):
        appearance_support(app(), {"appearance_evidence": "votes"})
