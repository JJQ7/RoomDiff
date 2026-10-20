"""Tests for scripts/exp_glob3r.py (the Glob3R experiment harness): resampling, the valid
mask, the depth fallback, frame checks, a synthetic round trip and the downstream runner."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from roomdiff.config import load_config
from roomdiff.eval_scenediff import run_state
from roomdiff.recon import load_recon

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp = load_script("exp_glob3r")


def no_edges(local, mask, rtol):
    return np.zeros(mask.shape, bool)


# ------------------------------------------------------------------ resampling


def test_rescale_intrinsics_keeps_pixel_rays_and_round_trips():
    K = np.array([[[5.0, 0.0, 3.7], [0.0, 6.0, 2.4], [0.0, 0.0, 1.0]]], np.float32)
    up = exp.rescale_intrinsics(K, (6, 8), (9, 12))  # x1.5 both ways
    assert up[0, 0, 0] == pytest.approx(7.5) and up[0, 1, 1] == pytest.approx(9.0)
    point = np.array([0.3, -0.2, 2.0])
    for axis in (0, 1):  # u + 0.5 scales with the image (pixel centres at integers)
        src = K[0, axis, axis] * point[axis] / point[2] + K[0, axis, 2]
        dst = up[0, axis, axis] * point[axis] / point[2] + up[0, axis, 2]
        assert dst + 0.5 == pytest.approx(1.5 * (src + 0.5), abs=1e-5)
    np.testing.assert_allclose(exp.rescale_intrinsics(up, (9, 12), (6, 8)), K, atol=1e-5)
    aniso = exp.rescale_intrinsics(K, (6, 8), (3, 16))  # y halves, x doubles
    assert aniso[0, 0, 0] == pytest.approx(10.0) and aniso[0, 1, 1] == pytest.approx(3.0)


def test_resize_nearest_copies_pixels_without_mixing():
    a = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    np.testing.assert_array_equal(exp.resize_nearest(a, 4, 6),
                                  np.repeat(np.repeat(a, 2, 1), 2, 2))
    b = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
    np.testing.assert_array_equal(exp.resize_nearest(b, 2, 3), b[:, [1, 3]][:, :, [1, 3, 5]])
    step = np.where(np.arange(16) < 7, 1.0, 3.0)[None, None].repeat(12, 1)  # (1, 12, 16)
    out = exp.resize_nearest(step, 9, 12)
    assert out.shape == (1, 9, 12) and set(np.unique(out)) <= {1.0, 3.0}
    stack = np.random.default_rng(0).random((2, 5, 7, 3))  # trailing channels kept
    assert exp.resize_nearest(stack, 10, 14).shape == (2, 10, 14, 3)


def test_resize_and_rescale_agree_within_half_a_source_pixel():
    """The source pixel sampled for an output pixel lies within half a source pixel of
    the continuous source position that the rescaled K assigns to that output pixel."""
    src_hw, dst_hw = (12, 16), (9, 13)
    K = np.array([[[10.0, 0.0, 7.2], [0.0, 10.0, 5.6], [0.0, 0.0, 1.0]]], np.float32)
    K2 = exp.rescale_intrinsics(K, src_hw, dst_hw)
    v, u = np.mgrid[0:src_hw[0], 0:src_hw[1]]
    sampled_u = exp.resize_nearest(u[None].astype(float), *dst_hw)[0]
    sampled_v = exp.resize_nearest(v[None].astype(float), *dst_hw)[0]
    dv, du = np.mgrid[0:dst_hw[0], 0:dst_hw[1]]
    x = (du - K2[0, 0, 2]) / K2[0, 0, 0] * K[0, 0, 0] + K[0, 0, 2]  # continuous source u
    y = (dv - K2[0, 1, 2]) / K2[0, 1, 1] * K[0, 1, 1] + K[0, 1, 2]
    assert np.abs(x - sampled_u).max() <= 0.5 + 1e-5
    assert np.abs(y - sampled_v).max() <= 0.5 + 1e-5


# ------------------------------------------------------------------ valid mask


def test_valid_mask_needs_conf_and_finite_positive_depth():
    depth = np.array([[[1.0, 0.0, np.nan], [np.inf, -1.0, 2.0]]], np.float32)
    conf = np.array([[[0.9, 0.9, 0.9], [0.9, 0.9, 0.1]]], np.float32)
    K = np.array([[[2.0, 0, 1.0], [0, 2.0, 0.5], [0, 0, 1]]], np.float32)
    seen = {}

    def spy(local, mask, rtol):
        seen.update(local=local, mask=mask, rtol=rtol)
        return np.zeros(mask.shape, bool)

    valid = exp.valid_mask(depth, K, conf, 0.25, 0.03, spy)
    expected = np.array([[[True, False, False], [False, False, False]]])
    np.testing.assert_array_equal(valid, expected)
    np.testing.assert_array_equal(seen["mask"], expected)  # edges only among good pixels
    assert np.isfinite(seen["local"]).all() and seen["rtol"] == 0.03
    np.testing.assert_allclose(seen["local"][0, 0, 0], [-0.5, -0.25, 1.0])  # K^-1 (u,v,1) z

    def edge_at_first(local, mask, rtol):
        edge = np.zeros(mask.shape, bool)
        edge[0, 0, 0] = True
        return edge

    assert not exp.valid_mask(depth, K, conf, 0.25, 0.03, edge_at_first).any()


def test_valid_mask_with_pi3_edges_drops_a_depth_step():
    pytest.importorskip("pi3.utils.geometry")
    depth = np.full((1, 16, 16), 2.0, np.float32)
    depth[:, :, 8:] = 1.0  # a step between two fronto-parallel planes
    K = np.array([[[20.0, 0, 7.5], [0, 20.0, 7.5], [0, 0, 1]]], np.float32)
    valid = exp.valid_mask(depth, K, np.ones_like(depth), 0.25, 0.03)
    assert valid[0, 2:-2, 2:6].all() and valid[0, 2:-2, 10:14].all()
    assert not valid[0, 2:-2, 7:9].any()


# ------------------------------------------------------------------ fallback


def test_fill_fallback_scales_the_baseline_depth():
    rng = np.random.default_rng(1)
    base_depth = rng.uniform(1.0, 3.0, (3, 20, 24)).astype(np.float32)
    base_valid = np.ones((3, 20, 24), bool)
    base_valid[:, :2] = False
    base = {"depth": base_depth, "valid": base_valid,
            "conf": rng.uniform(0.3, 1.0, (3, 20, 24)).astype(np.float16)}
    depth = np.zeros_like(base_depth)
    depth[0], depth[1] = 2.0 * base_depth[0], 3.0 * base_depth[1]
    depth[0, :2] = 100.0  # baseline-invalid pixels: ignored by the ratio
    depth[1, 5, :] = np.nan  # unknown raw depth: ignored
    depth[1, 6, :] = 0.0
    conf = np.full_like(base_depth, 0.7)
    has_depth = np.array([True, True, False])
    out_depth, out_conf, ratio, used = exp.fill_fallback(depth, conf, has_depth, base)
    assert used == 2 and ratio == pytest.approx(2.5)  # median of the frame ratios 2 and 3
    np.testing.assert_allclose(out_depth[2], 2.5 * base_depth[2], rtol=1e-6)
    np.testing.assert_array_equal(out_conf[2], base["conf"][2].astype(np.float32))
    np.testing.assert_array_equal(out_depth[:2], depth[:2])  # frames with depth untouched
    np.testing.assert_array_equal(out_conf[:2], conf[:2])
    assert depth[2].max() == 0.0  # the input is not modified

    ratio, used = exp.scale_ratio(depth, base_depth, base_valid, [0])
    assert used == 1 and ratio == pytest.approx(2.0)
    with pytest.raises(ValueError, match="no Glob3R depth"):
        exp.fill_fallback(np.zeros_like(base_depth), conf, np.zeros(3, bool), base)


# ------------------------------------------------------------------ frame checks


def test_check_frames_requires_baseline_order_and_split():
    base = {"files": np.array(["before_00000.jpg", "before_00001.jpg", "after_00000.jpg"]),
            "rec": np.array([0, 0, 1], np.uint8)}
    exp.check_frames({"files": base["files"], "rec": base["rec"]}, base)
    prefixed = np.array(["before/before_00000.jpg", "before/before_00001.jpg",
                         "after/after_00000.jpg"])
    exp.check_frames({"files": prefixed, "rec": np.array([0, 0, 1])}, base)
    with pytest.raises(ValueError, match="differ from the baseline at 2 positions"):
        exp.check_frames({"files": base["files"][[1, 0, 2]], "rec": base["rec"]}, base)
    with pytest.raises(ValueError, match="`rec` differs"):
        exp.check_frames({"files": base["files"], "rec": np.array([0, 1, 1])}, base)
    with pytest.raises(ValueError, match="2 frames"):
        exp.check_frames({"files": base["files"][:2], "rec": base["rec"][:2]}, base)


def test_check_raw_reports_missing_keys_shapes_and_poses():
    n, h, w = 2, 4, 5
    raw = {"files": np.array(["a.jpg", "b.jpg"]), "rec": np.array([0, 1], np.uint8),
           "cam2world": np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
           "K": np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
           "depth": np.ones((n, h, w), np.float32), "conf": np.ones((n, h, w), np.float32),
           "has_depth": np.ones(n, bool), "scale": np.ones(n, np.float32)}
    exp.check_raw(raw)
    with pytest.raises(ValueError, match="lacks"):
        exp.check_raw({k: v for k, v in raw.items() if k != "has_depth"})
    with pytest.raises(ValueError, match="conf"):
        exp.check_raw({**raw, "conf": np.ones((n, h + 1, w), np.float32)})
    poses = raw["cam2world"].copy()
    poses[1, 0, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite cam2world"):
        exp.check_raw({**raw, "cam2world": poses})
    poses[1, 0, 3] = 0.0
    poses[1, :3, :3] *= 1.3  # a similarity, not SE(3)
    with pytest.raises(ValueError, match=r"b\.jpg is not a rigid"):
        exp.check_raw({**raw, "cam2world": poses})


# ------------------------------------------------------------------ round trip


def synthetic_baseline(h: int = 36, w: int = 48) -> dict:
    """Two before and two after cameras (identity rotation, OpenCV axes: y down) seeing a
    wall at z = 1 and the floor at y = 0.15; both recordings see the same geometry."""
    K = np.array([[60.0, 0, (w - 1) / 2], [0, 60.0, (h - 1) / 2], [0, 0, 1]], np.float32)
    centres = [(-0.03, 0, 0), (0.03, 0, 0.03), (-0.02, 0.01, 0.02), (0.02, 0, -0.02)]
    v = np.mgrid[0:h, 0:w][0].astype(np.float64)
    dy = (v - K[1, 2]) / K[1, 1]
    poses, depths = [], []
    for c in centres:
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = c
        wall = np.full((h, w), 1.0 - c[2])
        with np.errstate(divide="ignore"):
            floor = np.where(dy > 1e-6, (0.15 - c[1]) / dy, np.inf)
        poses.append(T)
        depths.append(np.minimum(wall, floor))
    n = len(centres)
    rng = np.random.default_rng(0)
    return {"rec": np.array([0, 0, 1, 1], np.uint8),
            "files": np.array(["before_00000.jpg", "before_00001.jpg", "after_00000.jpg",
                               "after_00001.jpg"]),
            "images": rng.integers(0, 255, (n, h, w, 3), dtype=np.uint8),
            "cam2world": np.stack(poses), "K": np.tile(K, (n, 1, 1)),
            "depth": np.stack(depths).astype(np.float32),
            "conf": np.full((n, h, w), 0.9, np.float16), "valid": np.ones((n, h, w), bool),
            "metric_scale": 0.5}


@pytest.mark.parametrize("drop_every", [0, 2])
def test_from_baseline_round_trip(tmp_path, drop_every):
    base = synthetic_baseline()
    base_meta = {"diagnostics": {"verdict": "alignment OK", "frac_lt_3cm_min": 1.0}}
    params = load_config()["recon"]
    raw, raw_meta = exp.synthesise_raw(base, 0.75, drop_every)
    assert raw["depth"].shape == (4, 27, 36) and raw_meta["resolution"] == [36, 27]
    meta = exp.convert(raw, raw_meta, base, base_meta, params, tmp_path, no_edges)

    recon = load_recon(tmp_path / "recon.npz")
    assert recon["depth"].shape == base["depth"].shape and recon["conf"].dtype == np.float16
    np.testing.assert_allclose(recon["K"], base["K"], atol=1e-4)  # exact K round trip
    np.testing.assert_array_equal(recon["images"], base["images"])
    np.testing.assert_array_equal(recon["files"], base["files"])
    close = np.isclose(recon["depth"], base["depth"], rtol=0.02)
    assert close.mean() > 0.9  # nearest neighbour moves only pixels at the wall / floor seam
    assert meta["fallback_frames"] == (2 if drop_every else 0)
    assert meta["scale_ratio_vs_pi3x"] == pytest.approx(1.0, abs=0.02)
    assert meta["conf_source"].startswith("glob3r") and meta["source"] == "glob3r"
    assert meta["metric_scale"] == 0.5 and meta["levelled"] is True
    assert meta["level"]["method"] == "ransac floor plane"
    assert meta["baseline_diagnostics"] == base_meta["diagnostics"]
    assert meta["diagnostics"]["verdict"] == "alignment OK"
    assert meta["diagnostics"]["frac_lt_3cm_min"] > 0.95
    assert (tmp_path / "before.ply").is_file() and (tmp_path / "after.ply").is_file()
    floor_y = np.percentile(exp.unproject_depth(recon["depth"][0], recon["K"][0],
                                                recon["cam2world"][0])[..., 1], 1)
    assert abs(floor_y) < 0.02  # levelled: the floor at y = 0


def test_convert_brings_glob3r_units_back_to_the_pi3x_scale(tmp_path):
    base = synthetic_baseline()
    raw, raw_meta = exp.synthesise_raw(base, 1.0)
    raw["depth"] = raw["depth"] * 3.0  # a normalised frame: 3 units per metre
    raw["cam2world"][:, :3, 3] *= 3.0
    raw["has_depth"][1] = False
    raw["depth"][1] = 0.0
    params = load_config()["recon"]
    meta = exp.convert(raw, raw_meta, base, {}, params, tmp_path / "m", no_edges)
    assert meta["scale_ratio_vs_pi3x"] == pytest.approx(3.0, rel=1e-4)
    assert meta["matched_pi3x_scale"] and meta["fallback_frames"] == 1
    np.testing.assert_allclose(load_recon(tmp_path / "m" / "recon.npz")["depth"],
                               base["depth"], rtol=1e-4)
    assert meta["diagnostics"]["frac_lt_3cm_min"] > 0.95
    kept = exp.convert(raw, raw_meta, base, {}, params, tmp_path / "k", no_edges, False)
    assert not kept["matched_pi3x_scale"]
    np.testing.assert_allclose(load_recon(tmp_path / "k" / "recon.npz")["depth"],
                               3.0 * base["depth"], rtol=1e-4)


def test_convert_uses_pi3x_conf_when_glob3r_gives_none(tmp_path):
    base = synthetic_baseline()
    raw, raw_meta = exp.synthesise_raw(base, 1.0)
    raw["conf"] = np.ones_like(raw["conf"])
    raw["files"] = raw["files"][[1, 0, 2, 3]]
    params = load_config()["recon"]
    with pytest.raises(ValueError, match="differ from the baseline"):
        exp.convert(raw, raw_meta, base, {}, params, tmp_path, no_edges)
    raw["files"] = base["files"]
    meta = exp.convert(raw, raw_meta, base, {}, params, tmp_path, no_edges)
    assert meta["conf_source"].startswith("pi3x")
    np.testing.assert_array_equal(load_recon(tmp_path / "recon.npz")["conf"], base["conf"])


# ------------------------------------------------------------------ downstream / score


def make_run(root: Path) -> Path:
    run = root / "runs" / "pair_a"
    for d in ("keyframes", "recon", "glob3r", "visibility", "appearance", "instances",
              "semantics", "crops", "viewer"):
        (run / d).mkdir(parents=True)
        (run / d / "x").write_text("x")
    (run / "recon" / "recon.npz").write_text("x")
    for f in ("changes.json", "report.md", "scene.rrd", "changes.rrd"):
        (run / f).write_text("{}")
    (run / "recon_pi3x").symlink_to(run / "recon", target_is_directory=True)
    (run / "eval_run.json").write_text(json.dumps({"rc": 0}))
    return run


def test_remove_stale_keeps_inputs_and_earlier_stages(tmp_path):
    run = make_run(tmp_path)
    removed = exp.remove_stale(run, ["describe"])
    assert set(removed) == {"semantics", "report.md", "viewer", "eval_run.json", "scene.rrd",
                            "changes.rrd"}
    assert (run / "changes.json").is_file() and (run / "visibility").is_dir()
    exp.remove_stale(run, list(exp.STAGES))
    assert sorted(p.name for p in run.iterdir()) == ["glob3r", "keyframes", "recon",
                                                     "recon_pi3x"]
    assert (run / "recon_pi3x" / "recon.npz").is_file()


def test_downstream_pair_stops_at_the_first_failure(tmp_path):
    run = make_run(tmp_path)
    calls, failing = [], {"instances"}

    def runner(cmd, log):
        stage = cmd[cmd.index("roomdiff.cli") + 1]
        calls.append(stage)
        if stage == "changes":
            (run / "changes.json").write_text("{}")
        return 3 if stage in failing else 0

    log = tmp_path / "logs" / "pair_a.log"
    marker = exp.downstream_pair(run, log, exp.STAGES, runner=runner)
    assert calls == ["visibility", "appearance", "instances"]
    assert marker["rc"] == 3 and marker["failed_stage"] == "instances"
    assert json.loads((run / "eval_run.json").read_text())["rc"] == 3
    assert run_state(run) == "failed" and not (run / "visibility").exists()

    failing.clear()
    marker = exp.downstream_pair(run, log, exp.STAGES, ["visibility.unverified_floor_cut=0"],
                                 runner=runner)
    assert marker["rc"] == 0 and run_state(run) == "done"
    assert calls[3:] == list(exp.STAGES)
    cmd = marker["cmd"][0]
    assert cmd[:7] == [str(exp.GUARD), "-g", "-m", "10G", "-t", "1500", "--"]
    assert cmd[7:] == [str(exp.PYTHON), "-m", "roomdiff.cli", "visibility", "--out", str(run),
                       "--set", "ingest.mode=fixed", "--set", "ingest.fixed_interval_s=0.5",
                       "--set", "visibility.unverified_floor_cut=0"]


def test_score_pairs_only_takes_finished_converted_runs(tmp_path):
    def run(name, recon=True, marker=None, changes=True):
        d = tmp_path / "runs" / name
        (d / "recon").mkdir(parents=True)
        if recon:
            (d / "recon" / "recon.npz").write_text("x")
        if marker is not None:
            (d / "eval_run.json").write_text(json.dumps(marker))
        if changes:
            (d / "changes.json").write_text("{}")

    run("ok", marker={"rc": 0})
    run("failed", marker={"rc": 137, "failed_stage": "describe"})
    run("not_run")
    run("no_recon", recon=False, marker={"rc": 0})
    used, excluded = exp.score_pairs(tmp_path, ["ok", "failed", "not_run", "no_recon"])
    assert used == ["ok"]
    assert excluded == {"failed": "downstream failed (rc 137, describe)",
                        "not_run": "no downstream run", "no_recon": "no converted recon"}


def test_alignment_rows_count_verdict_changes():
    def meta(verdict, frac):
        return {"diagnostics": {"verdict": verdict, "frac_lt_3cm_min": frac,
                                "nn_covisible": {}, "nn_all": {}}}

    rows = [exp.alignment_row("a", meta("alignment WEAK", 0.5), meta("alignment FAIR", 0.7)),
            exp.alignment_row("b", meta("alignment FAIR", 0.75), meta("alignment FAIR", 0.73)),
            exp.alignment_row("c", meta("alignment OK", 0.9), meta("alignment WEAK", 0.6)),
            exp.alignment_row("d", meta("alignment FAIR", 0.7), None)]
    summary = exp.alignment_summary(rows)
    assert (summary["improved"], summary["unchanged"], summary["worsened"]) == (1, 1, 1)
    assert summary["converted"] == 3
    assert summary["mean_delta_frac_lt_3cm_min"] == pytest.approx((0.2 - 0.02 - 0.3) / 3)
    table = exp.alignment_table(rows)
    assert "WEAK → FAIR" in table and "| d | FAIR |" in table
