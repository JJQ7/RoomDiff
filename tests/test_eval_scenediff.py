"""Tests for the SceneDiff evaluation path (CPU): benchmark frame mapping, the
object_masks.pkl format, and the official evaluator on a synthetic scene.

The synthetic scene is the one of tests/test_changes.py (red box moved, blue removed,
green added, yellow unverified). Its keyframe j of each recording stands for benchmark
frame j; the GT is built from the true label maps at twice the reconstruction resolution.
"""

import json
import os
import pickle
import shutil
import subprocess

import numpy as np
import pytest
from pycocotools import mask as mask_utils
from test_changes import SETTINGS, make_scene
from typer.testing import CliRunner

from roomdiff import eval_scenediff
from roomdiff.changes import run_changes
from roomdiff.cli import app
from roomdiff.config import load_config
from roomdiff.eval_scenediff import (
    EVALUATOR,
    GUARD,
    benchmark_frames,
    evaluate,
    load_segments,
    run_pairs,
    side_masks,
    write_prediction,
)
from roomdiff.instances import decode_rle, encode_rle, run_instances
from roomdiff.models import FakeFeaturizer, FakeSegmenter
from roomdiff.visibility import run_visibility

PAIR = "synthetic_1_synthetic_2"
N_VIEWS = 10  # keyframes per recording in make_scene
GT_SCALE = 2  # GT masks at twice the reconstruction resolution (the evaluator resizes)
pytestmark = pytest.mark.filterwarnings(  # pycocotools vs numpy 2, harmless
    "ignore:__array__ implementation:DeprecationWarning")
GT_OBJECTS = [  # (label, label-map id, in video 1, in video 2)
    ("red box", 1, True, True), ("blue box", 2, True, False), ("green box", 3, False, True)]


def write_index(out) -> None:
    """keyframes/index.json as fixed-mode ingest writes it (keyframe j = time j s)."""
    index = {}
    for v, rec in enumerate(("before", "after"), start=1):
        frames = [{"file": f"{rec}_{j:05d}.jpg", "frame_index": 30 * j, "timestamp": j + 0.01,
                   "sharpness": 100.0, "width": 320, "height": 240, "t_target": float(j),
                   "benchmark_frame": j} for j in range(N_VIEWS)]
        index[rec] = {"video": {"path": f"/data/{PAIR}/original_video{v}.mp4"}, "frames": frames}
    (out / "keyframes").mkdir(parents=True, exist_ok=True)
    (out / "keyframes" / "index.json").write_text(json.dumps(index))


def write_gt(pair_dir, labels) -> None:
    """segments.pkl: masks at frames 30 j (kept by the evaluator) and 30 j + 15 (ignored)."""
    segments = {"scene_type": "test", "light_change": False, "comments": {}, "objects": [],
                "video1_objects": {}, "video2_objects": {}}
    for idx, (label, value, in1, in2) in enumerate(GT_OBJECTS):
        segments["objects"].append({"label": label, "in_video1": in1, "in_video2": in2,
                                    "original_obj_idx": idx})
        for key, offset, present in (("video1_objects", 0, in1), ("video2_objects", N_VIEWS, in2)):
            if not present:
                continue
            tracks = segments[key].setdefault(idx, {})
            for j in range(N_VIEWS):
                mask = labels[str(offset + j)] == value
                if mask.any():
                    big = mask.repeat(GT_SCALE, 0).repeat(GT_SCALE, 1)
                    tracks[str(30 * j)] = tracks[str(30 * j + 15)] = encode_rle(big)
    pair_dir.mkdir(parents=True)
    with open(pair_dir / "segments.pkl", "wb") as f:
        pickle.dump(segments, f, protocol=4)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """(eval root with runs/<PAIR>/, benchmark root with data/<PAIR>/, changes doc)."""
    tmp = tmp_path_factory.mktemp("scenediff")
    out = tmp / "eval" / "runs" / PAIR
    out.mkdir(parents=True)
    labels = make_scene(out)
    cfg = load_config(overrides=SETTINGS)
    run_visibility(out, cfg)
    run_instances(out, cfg, FakeSegmenter(labels), FakeFeaturizer())
    doc = run_changes(out, cfg)
    write_index(out)
    write_gt(tmp / "benchmark" / "data" / PAIR, labels)
    return tmp / "eval", tmp / "benchmark", doc


# ------------------------------------------------------------------ units


def test_benchmark_frames_mapping():
    frames = [{"file": "a", "timestamp": 0.05}, {"file": "b", "timestamp": 0.95},
              {"file": "c", "timestamp": 1.04}, {"file": "d", "timestamp": 1.5},
              {"file": "e", "timestamp": 2.12},
              {"file": "f", "timestamp": 3.4, "benchmark_frame": 3},
              {"file": "g", "timestamp": 3.5, "benchmark_frame": None}]
    # b and c both round to 1: the closer one (c) wins; d, e are too far from a second;
    # fixed-mode keyframes use benchmark_frame whatever their timestamp, null = none.
    assert benchmark_frames(frames, 0.1) == {"a": 0, "c": 1, "f": 3}


def test_side_masks_unite_the_parts_of_a_merged_object():
    seat, base = np.zeros((20, 30), bool), np.zeros((20, 30), bool)
    seat[2:8, 5:20] = True
    base[10:18, 8:12] = True
    per_instance = {1: {0: (0.9, encode_rle(seat)), 3: (0.8, encode_rle(seat))},
                    2: {0: (0.7, encode_rle(base)), 5: (0.6, encode_rle(base))}}
    out = side_masks(per_instance, [1, 2])
    assert sorted(out) == [0, 3, 5]
    assert out[0][0] == 0.9 and np.array_equal(decode_rle(out[0][1]), seat | base)
    assert out[3] == per_instance[1][3] and out[5] == per_instance[2][5]
    assert side_masks(per_instance, [1]) == per_instance[1] and side_masks(per_instance, [9]) == {}


def test_load_segments_refuses_globals(tmp_path):
    path = tmp_path / "segments.pkl"
    path.write_bytes(pickle.dumps({"objects": [], "hook": os.getcwd}))
    with pytest.raises(pickle.UnpicklingError, match="refusing global"):
        load_segments(path)
    data = {"objects": [{"label": "x", "in_video1": True}], "video1_objects": {0: {"0": {}}}}
    path.write_bytes(pickle.dumps(data))
    assert load_segments(path) == data


# ------------------------------------------------------------------ predictions


def test_write_prediction_matches_the_evaluator_format(synthetic):
    eval_root, bench, doc = synthetic
    pred_dir = eval_root / "pred_format" / PAIR
    stats = write_prediction(eval_root / "runs" / PAIR, bench / "data" / PAIR, pred_dir)
    with open(pred_dir / "object_masks.pkl", "rb") as f:
        pred = pickle.load(f)
    assert (pred["H"], pred["W"]) == (768, 1024)  # GT 480 x 640 scaled to max side 1024
    changes = {c["id"]: c for c in doc["changes"]}
    objects = {k: v for k, v in pred.items() if k not in ("H", "W")}
    assert {changes[k]["type"] for k in objects} >= {"moved", "removed", "added"}
    for obj_id, obj in objects.items():
        change = changes[obj_id]
        videos = set(obj) - {"label"}
        expected = {"moved": {"video_1", "video_2"}, "removed": {"video_1"},
                    "added": {"video_2"}}.get(change["type"])
        assert videos == expected if expected else videos in ({"video_1"}, {"video_2"})
        assert obj["label"] == (1 if change["type"] == "moved" else 0)
        for video in videos:
            for k, entry in obj[video].items():
                assert isinstance(k, int) and 0 <= k < N_VIEWS
                mask = mask_utils.decode(entry["mask"])  # exactly as the evaluator does
                assert mask.shape == (240, 320) and mask.any()
                assert entry["cost"] == pytest.approx(change["confidence"] * 0.9, rel=1e-5)
    assert stats["objects"] == len(objects) and stats["aspect_ok"] is True
    assert stats["benchmark_frames"]["video_1"] == list(range(N_VIEWS))

    verified = write_prediction(eval_root / "runs" / PAIR, bench / "data" / PAIR,
                                eval_root / "pred_verified" / PAIR, include_unverified=False)
    assert "unverified" not in verified["by_type"] and verified["objects"] == 3


def test_write_prediction_rejects_swapped_videos(synthetic, tmp_path):
    eval_root, bench, _ = synthetic
    index_file = eval_root / "runs" / PAIR / "keyframes" / "index.json"
    original = index_file.read_text()
    index = json.loads(original)
    before, after = index["before"]["video"], index["after"]["video"]
    index["before"]["video"], index["after"]["video"] = after, before
    try:
        index_file.write_text(json.dumps(index))
        with pytest.raises(ValueError, match="original_video1"):
            write_prediction(eval_root / "runs" / PAIR, bench / "data" / PAIR, tmp_path / PAIR)
    finally:
        index_file.write_text(original)


def test_run_pairs_runs_guarded_and_retries_failures(synthetic, tmp_path, monkeypatch):
    eval_root, bench, _ = synthetic
    pair_dir = tmp_path / "bench" / "data" / PAIR
    pair_dir.mkdir(parents=True)
    shutil.copy(bench / "data" / PAIR / "segments.pkl", pair_dir)
    for name in ("original_video1.mp4", "original_video2.MOV", "video1.mp4"):
        (pair_dir / name).touch()
    calls = []

    def failing_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 3)

    monkeypatch.setattr(eval_scenediff.subprocess, "run", failing_run)
    cfg = load_config(overrides=[f"eval.data_root={tmp_path / 'bench'}"])
    results = run_pairs([PAIR], tmp_path / "out", cfg, overrides=["recon.max_frames=50"])
    cmd = calls[0]
    assert cmd[:2] == [str(GUARD), "-g"] and cmd[cmd.index("-m") + 1] == "10G"
    assert cmd[cmd.index("--before") + 1] == str(pair_dir / "original_video1.mp4")
    assert cmd[cmd.index("--after") + 1] == str(pair_dir / "original_video2.MOV")
    assert "ingest.mode=fixed" in cmd and cmd[-2:] == ["--set", "recon.max_frames=50"]
    assert results[PAIR]["run"] == "failed" and results[PAIR]["objects"] == 0
    with open(tmp_path / "out" / "pred" / PAIR / "object_masks.pkl", "rb") as f:
        assert set(pickle.load(f)) == {"H", "W"}  # empty: the GT objects count as misses
    run_pairs([PAIR], tmp_path / "out", cfg)
    assert len(calls) == 2  # a failed run is retried
    assert run_pairs([PAIR], tmp_path / "out", cfg, skip_run=True)[PAIR]["run"] == "failed"

    shutil.copytree(eval_root / "runs" / PAIR, tmp_path / "done" / "runs" / PAIR)
    results = run_pairs([PAIR], tmp_path / "done", cfg)  # finished run: not re-run
    assert len(calls) == 2 and results[PAIR]["run"] == "done" and results[PAIR]["objects"] >= 3


# ------------------------------------------------------------------ official evaluator


@pytest.mark.skipif(not EVALUATOR.is_file(), reason="third_party/scene_diff is not checked out")
def test_cli_eval_runs_the_official_evaluator(synthetic):
    eval_root, bench, _ = synthetic
    result = CliRunner().invoke(app, ["eval", "--pairs", PAIR, "--out-root", str(eval_root),
                                      "--skip-run", "-o", f"eval.data_root={bench}"])
    assert result.exit_code == 0, result.output
    summary = json.loads((eval_root / "pred" / "summary.json").read_text())
    m = summary["metrics"]
    assert summary["not_predicted"] == [] and summary["evaluator_errors"] == []
    assert m["scenes"] == 1 and m["object_tp"] == 3 and m["object_fn"] == 0
    assert m["per_scene_ap"] == pytest.approx(1.0) and m["per_scene_ap_type"] == pytest.approx(1.0)
    assert m["per_view_ap"] > 0.5 and m["global_iou"] > 0.5


@pytest.mark.skipif(not EVALUATOR.is_file(), reason="third_party/scene_diff is not checked out")
def test_evaluate_reports_pairs_without_prediction(synthetic):
    eval_root, bench, _ = synthetic
    pred_root = eval_root / "pred_missing"
    write_prediction(eval_root / "runs" / PAIR, bench / "data" / PAIR, pred_root / PAIR)
    summary = evaluate(pred_root, [PAIR, "not_run_pair"], root=bench)
    assert summary["not_predicted"] == ["not_run_pair"] and summary["metrics"]["scenes"] == 1
    assert (pred_root / "_evaluator" / "gt" / PAIR / "segments.pkl").is_file()


def test_cli_eval_rejects_unknown_pairs(tmp_path):
    result = CliRunner().invoke(app, ["eval", "--pairs", "no_such_pair", "--out-root",
                                      str(tmp_path), "--skip-run"])
    assert result.exit_code == 2 and "no_such_pair" in result.output
