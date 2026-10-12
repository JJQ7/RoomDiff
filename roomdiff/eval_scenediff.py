"""SceneDiff benchmark evaluation: our changes as the official `object_masks.pkl` and the
official evaluator (`third_party/scene_diff/scripts/evaluate_multiview.py`, MIT licence).

Frames: the evaluator keeps ground-truth (GT) frames whose index in the benchmark's 30 fps
re-encode is a multiple of 30 and renumbers them k = index // 30, so GT frame k shows time
k s of the original video. A keyframe stands for frame k if it carries `benchmark_frame`
= k (fixed ingest mode, `ingest.mode=fixed`) or, for sharp-mode keyframes, if its
timestamp is within `eval.time_tol_s` of k; when several keyframes map to one k the
closest in time wins. Recording `before` = `video_1`, `after` = `video_2`.

Prediction per change (object id = change id): its before-instance masks at the mapped
keyframes -> `video_1` entries, after-instance masks -> `video_2`; `cost` = change
confidence x SAM mask score; `label` 1 for moved, else 0 (the evaluator recomputes it from
the presence of both video keys, so a moved change keeps both keys even if one is empty).
Masks are COCO RLE at the reconstruction resolution (upright, same aspect as the GT); the
evaluator resizes them to its working size. Layout below `out_root`: `runs/<pair>/`
(pipeline outputs), `logs/<pair>.log`, `pred/<pair>/object_masks.pkl` and
`prediction.json`, `pred/summary.json`, `pred/result_frame_dup_1_obj_dup_1.txt`.

Evaluator quirks handled here: it imports `faiss` through `utils.py` (only needed for
visualisation; stubbed by the launcher), reads the split lists from the relative path
`data/scenediff_benchmark/splits/` (we run it in a work directory holding split files that
list exactly our pairs), loads `segments.pkl` with plain `pickle.load` (we give it copies
re-pickled from a data-only unpickle of the untrusted originals), and silently drops
scenes without a prediction file (failed runs therefore get an empty prediction).
"""

from __future__ import annotations

import json
import pickle
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from roomdiff.instances import load_masks
from roomdiff.paths import RECORDINGS, OutputPaths

ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = ROOT / "third_party" / "scene_diff" / "scripts" / "evaluate_multiview.py"
GUARD = ROOT / "scripts" / "run_guarded.sh"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON if VENV_PYTHON.is_file() else Path(sys.executable))
MAX_LENGTH = 1024  # the evaluator's default --max_length
VIDEO_KEYS = {"before": "video_1", "after": "video_2"}
VIDEO_STEMS = {"before": "original_video1", "after": "original_video2"}
RESULT_SUFFIX = "_frame_dup_1_obj_dup_1"  # appended by the evaluator (default thresholds)
FLOAT_METRICS = ("global_iou", "per_view_ap", "per_scene_ap", "per_scene_ap_type")
RUN_MARKER = "eval_run.json"  # written by run_pipeline: return code and command

# Runs the official script unmodified. Its `from utils import process_video_to_frames`
# would import third_party/scene_diff/utils.py, which needs faiss (not installed) and is
# only used for --visualize True: a stub module stands in for it.
LAUNCHER = """\
import runpy, sys, types
def process_video_to_frames(*args, **kwargs):
    raise RuntimeError("visualisation needs third_party/scene_diff/utils.py (faiss)")
stub = types.ModuleType("utils")
stub.process_video_to_frames = process_video_to_frames
sys.modules["utils"] = stub
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


# ------------------------------------------------------------------ dataset


class _DataOnlyUnpickler(pickle.Unpickler):
    """Refuses every global (class or function), so only plain data can be loaded."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"refusing global {module}.{name} in an untrusted pickle")


def load_segments(path: str | Path) -> dict[str, Any]:
    """Load a (untrusted) SceneDiff `segments.pkl`: dicts, lists, strings, numbers only."""
    with open(path, "rb") as f:
        data = _DataOnlyUnpickler(f).load()
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a dict at the top level")
    return data


def data_root(cfg: Mapping[str, Any]) -> Path:
    """`eval.data_root` (relative paths are relative to the repository root)."""
    path = Path(cfg["eval"]["data_root"])
    return path if path.is_absolute() else ROOT / path


def select_pairs(root: Path, split: str, subset: str = "varied", limit: int | None = None
                 ) -> list[str]:
    """Pair names of `splits/<split>_split.json` (subset varied / kitchen / all)."""
    doc = json.loads((root / "splits" / f"{split}_split.json").read_text())
    names = doc["varied"] + doc["kitchen"] if subset == "all" else list(doc[subset])
    return names[:limit] if limit else names


def pair_videos(pair_dir: Path) -> tuple[Path, Path]:
    """`original_video1.*` and `original_video2.*` (raw phone videos) of a pair folder."""
    found = []
    for stem in VIDEO_STEMS.values():
        matches = sorted(pair_dir.glob(f"{stem}.*"))
        if not matches:
            raise FileNotFoundError(f"no {stem}.* in {pair_dir}")
        found.append(matches[0])
    return found[0], found[1]


def gt_mask_size(segments: Mapping[str, Any]) -> tuple[int, int] | None:
    """Largest `[H, W]` of the GT RLEs (upright re-encode resolution), None if no masks."""
    sizes = [tuple(int(s) for s in rle["size"])
             for key in ("video1_objects", "video2_objects")
             for frames in segments.get(key, {}).values() for rle in frames.values()]
    return max(sizes, key=lambda s: s[0] * s[1]) if sizes else None


def scaled_size(height: int, width: int, max_length: int = MAX_LENGTH) -> tuple[int, int]:
    """`(H, W)` scaled so that the longer side is `max_length`."""
    scale = max_length / max(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


# ------------------------------------------------------------------ predictions


def benchmark_frames(frames: Sequence[Mapping[str, Any]], time_tol: float) -> dict[str, int]:
    """Keyframe file -> GT frame k for one recording (entries of keyframes/index.json).

    Fixed-mode keyframes use their `benchmark_frame` (null: none); others map to
    round(timestamp) if within `time_tol` s. One keyframe per k: the closest in time.
    """
    best: dict[int, tuple[float, str]] = {}
    for frame in frames:
        t = float(frame["timestamp"])
        if "benchmark_frame" in frame:
            k = frame["benchmark_frame"]
            if k is None:
                continue
        else:
            k = round(t)
            if abs(t - k) >= time_tol:
                continue
        error = abs(t - k)
        if k not in best or error < best[k][0]:
            best[k] = (error, frame["file"])
    return {file: k for k, (_, file) in best.items()}


def write_prediction(
    out_dir: str | Path, pair_dir: str | Path, pred_dir: str | Path,
    include_unverified: bool = True, time_tol: float = 0.1,
) -> dict[str, Any]:
    """Write `<pred_dir>/object_masks.pkl` (+ `prediction.json` stats) of one scene from a
    pipeline output `<out_dir>` (changes.json, instances/<rec>_masks.npz, keyframes/
    index.json). `pair_dir` (its segments.pkl) gives the GT size. Returns the stats."""
    paths, pred_dir = OutputPaths(out_dir), Path(pred_dir)
    doc = json.loads(paths.changes_json.read_text())
    index = json.loads(paths.keyframes_index.read_text())
    for rec, stem in VIDEO_STEMS.items():  # before must be video 1, after video 2
        video = index[rec].get("video") or {}
        if video.get("path") and Path(video["path"]).stem != stem:
            raise ValueError(f"{out_dir}: the {rec} video is {video['path']}, expected {stem}.*")

    masks: dict[str, dict[int, dict[int, tuple[float, dict[str, Any]]]]] = {}
    mapped: dict[str, list[int]] = {}
    mask_size: tuple[int, int] | None = None
    for rec in RECORDINGS:
        frame_of = benchmark_frames(index[rec]["frames"], time_tol)
        mapped[rec] = sorted(frame_of.values())
        per_instance: dict[int, dict[int, tuple[float, dict[str, Any]]]] = {}
        for file, entries in load_masks(paths.instances / f"{rec}_masks.npz").items():
            if file not in frame_of:
                continue
            for instance_id, score, rle in entries:
                mask_size = (int(rle["size"][0]), int(rle["size"][1]))
                frames = per_instance.setdefault(instance_id, {})
                if frame_of[file] not in frames or score > frames[frame_of[file]][0]:
                    frames[frame_of[file]] = (score, rle)
        masks[rec] = per_instance

    pred: dict[Any, Any] = {}
    kinds: Counter[str] = Counter()
    for change in doc["changes"]:
        if change["type"] == "unverified" and not include_unverified:
            continue
        obj: dict[str, Any] = {}
        for rec in RECORDINGS:
            if change[rec] is None:
                continue
            frames = masks[rec].get(change[rec]["instance_id"], {})
            obj[VIDEO_KEYS[rec]] = {
                k: {"mask": {"size": list(rle["size"]), "counts": rle["counts"]},
                    "cost": float(change["confidence"]) * score}
                for k, (score, rle) in sorted(frames.items())}
        if not any(obj.values()):  # no mask at any benchmark frame
            continue
        obj["label"] = 1 if change["type"] == "moved" else 0
        pred[change["id"]] = obj
        kinds[change["type"]] += 1

    segments = load_segments(Path(pair_dir) / "segments.pkl")
    gt_size = gt_mask_size(segments)
    height, width = scaled_size(*(gt_size or mask_size or (MAX_LENGTH, MAX_LENGTH)))
    pred["H"], pred["W"] = height, width
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / "object_masks.pkl", "wb") as f:
        pickle.dump(pred, f, protocol=4)

    aspect_ok = None
    if gt_size and mask_size:
        aspect_ok = abs(mask_size[0] / mask_size[1] / (gt_size[0] / gt_size[1]) - 1) < 0.03
        if not aspect_ok:
            print(f"[eval] WARNING {pair_dir}: mask size {mask_size} and GT size {gt_size} "
                  "differ in aspect ratio (orientation?)")
    stats = {
        "pair": Path(pair_dir).name, "run_dir": str(out_dir), "objects": sum(kinds.values()),
        "by_type": dict(kinds), "include_unverified": include_unverified,
        "masks": {VIDEO_KEYS[r]: sum(len(o.get(VIDEO_KEYS[r], {})) for k, o in pred.items()
                                     if k not in ("H", "W")) for r in RECORDINGS},
        "benchmark_frames": {VIDEO_KEYS[r]: mapped[r] for r in RECORDINGS},
        "mask_size": mask_size, "gt_size": gt_size, "target_size": [height, width],
        "aspect_ok": aspect_ok,
    }
    (pred_dir / "prediction.json").write_text(json.dumps(stats, indent=1))
    return stats


def write_empty_prediction(pair_dir: Path, pred_dir: Path, reason: str) -> dict[str, Any]:
    """Prediction without objects, so a failed run counts as missed GT objects."""
    gt_size = gt_mask_size(load_segments(pair_dir / "segments.pkl"))
    height, width = scaled_size(*(gt_size or (MAX_LENGTH, MAX_LENGTH)))
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / "object_masks.pkl", "wb") as f:
        pickle.dump({"H": height, "W": width}, f, protocol=4)
    stats = {"pair": pair_dir.name, "objects": 0, "failed": reason}
    (pred_dir / "prediction.json").write_text(json.dumps(stats, indent=1))
    return stats


# ------------------------------------------------------------------ pipeline runs


def run_state(out: Path) -> str:
    """Run state of `<out>`: done (changes.json, no failed run marker), failed, missing."""
    marker = out / RUN_MARKER
    if marker.is_file() and json.loads(marker.read_text())["rc"] != 0:
        return "failed"
    return "done" if OutputPaths(out).changes_json.is_file() else "missing"


def run_pipeline(
    pair_dir: Path, out: Path, log: Path, params: Mapping[str, Any],
    config_file: str | Path | None = None, overrides: Sequence[str] = (),
) -> str:
    """`roomdiff run --set ingest.mode=fixed` on one pair through scripts/run_guarded.sh
    (GPU lock, memory cap, timeout); writes `<out>/eval_run.json`. Returns run_state."""
    before, after = pair_videos(pair_dir)
    cmd = [str(GUARD), "-g", "-m", str(params["run_mem"]), "-t", str(params["run_timeout_s"]),
           "--", PYTHON, "-m", "roomdiff.cli", "run", "--before", str(before),
           "--after", str(after), "--out", str(out), "--set", "ingest.mode=fixed"]
    if config_file is not None:
        cmd += ["--config", str(Path(config_file).resolve())]
    for item in overrides:
        cmd += ["--set", item]
    start = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    out.mkdir(parents=True, exist_ok=True)
    (out / RUN_MARKER).write_text(json.dumps(
        {"rc": rc, "seconds": round(time.perf_counter() - start, 1), "cmd": cmd}, indent=1))
    return run_state(out)


def run_pairs(
    pairs: Sequence[str], out_root: str | Path, cfg: Mapping[str, Any],
    config_file: str | Path | None = None, overrides: Sequence[str] = (),
    skip_run: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run the pipeline on every pair that has no finished run in `runs/<pair>/` (one at a
    time, GPU-locked; none with `skip_run`) and write its prediction. A failed run gets an
    empty prediction (its GT objects count as misses) and is retried next time; a pair
    without a run (`skip_run`) gets no prediction, so the evaluator leaves it out."""
    params, root = cfg["eval"], data_root(cfg)
    out_root = Path(out_root).resolve()
    results = {}
    for i, pair in enumerate(pairs, start=1):
        pair_dir, out = root / "data" / pair, out_root / "runs" / pair
        pred_dir, log = out_root / "pred" / pair, out_root / "logs" / f"{pair}.log"
        state = run_state(out)
        if state != "done" and not skip_run:
            print(f"[eval] {i}/{len(pairs)} {pair}: running the pipeline (log {log})")
            start = time.perf_counter()
            state = run_pipeline(pair_dir, out, log, params, config_file, overrides)
            print(f"[eval] {i}/{len(pairs)} {pair}: {state} in "
                  f"{time.perf_counter() - start:.0f} s")
        if state == "missing":
            print(f"[eval] {pair}: no {out}/changes.json, left out")
            (pred_dir / "object_masks.pkl").unlink(missing_ok=True)  # stale earlier one
            continue
        if state == "failed":
            stats = write_empty_prediction(pair_dir, pred_dir, f"run failed, see {log}")
        else:
            stats = write_prediction(out, pair_dir, pred_dir, params["include_unverified"],
                                     params["time_tol_s"])
        results[pair] = {"run": state, **stats}
    return results


# ------------------------------------------------------------------ official evaluator


def parse_result(text: str) -> dict[str, float]:
    """Numbers of the evaluator's result file."""
    patterns = {
        "scenes": r"Scenes: (\d+)",
        "global_iou": r"Metric 1: px/im IoU[^:]*: ([\d.]+)",
        "pixel_tp": r"TP: (\d+), FP: \d+, FN: \d+",
        "pixel_fp": r"TP: \d+, FP: (\d+), FN: \d+",
        "pixel_fn": r"TP: \d+, FP: \d+, FN: (\d+)",
        "per_view_ap": r"Metric 2: obj/im AP[^:]*: ([\d.]+)",
        "per_scene_ap": r"Metric 3a: obj/sc AP[^:]*: ([\d.]+)",
        "per_scene_ap_type": r"Metric 3b: obj/sc AP[^:]*: ([\d.]+)",
        "object_tp": r"True Positives: (\d+)",
        "object_fp": r"False Positives: (\d+)",
        "object_fn": r"False Negatives: (\d+)",
    }
    out: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"evaluator result has no {key!r} ({pattern})")
        out[key] = float(match.group(1)) if key in FLOAT_METRICS else int(match.group(1))
    return out


def evaluate(
    pred_root: str | Path, pairs: Sequence[str], result_path: str | Path | None = None,
    root: str | Path = ROOT / "data" / "scenediff" / "scenediff_benchmark",
    timeout: float = 3600,
) -> dict[str, Any]:
    """Run the official evaluator on `<pred_root>/<pair>/object_masks.pkl` for `pairs` and
    write `<pred_root>/summary.json` (metrics, per-pair prediction stats, scenes the
    evaluator could not load). Returns the summary."""
    pred_root, root = Path(pred_root).resolve(), Path(root).resolve()
    result_path = Path(result_path or pred_root / "result.txt").resolve()
    work = pred_root / "_evaluator"
    splits = work / "data" / "scenediff_benchmark" / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "val_split.json").write_text(json.dumps({"varied": list(pairs), "kitchen": []}))
    (splits / "test_split.json").write_text(json.dumps({"varied": [], "kitchen": []}))
    gt_dir = work / "gt"
    predicted = [p for p in pairs if (pred_root / p / "object_masks.pkl").is_file()]
    for pair in predicted:  # data-only copies: the evaluator uses plain pickle.load
        (gt_dir / pair).mkdir(parents=True, exist_ok=True)
        with open(gt_dir / pair / "segments.pkl", "wb") as f:
            pickle.dump(load_segments(root / "data" / pair / "segments.pkl"), f, protocol=4)

    cmd = [PYTHON, "-I", "-c", LAUNCHER, str(EVALUATOR), "--pred_dir", str(pred_root),
           "--gt_dir", str(gt_dir), "--video_dir", str(root / "data"),
           "--output_path", str(result_path), "--visualize", "False",
           "--splits", "val", "--sets", "all"]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=timeout)
    log = pred_root / "evaluator_log.txt"
    log.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        raise RuntimeError(f"evaluator failed (rc {proc.returncode}, log {log}):\n{tail}")
    result_file = result_path.with_name(result_path.stem + RESULT_SUFFIX + result_path.suffix)

    stats = {}
    for pair in predicted:
        file = pred_root / pair / "prediction.json"
        stats[pair] = json.loads(file.read_text()) if file.is_file() else None
    summary = {
        "pairs": list(pairs),
        "not_predicted": [p for p in pairs if p not in predicted],
        "evaluator_errors": re.findall(r"^Error .*$", proc.stdout, flags=re.M),
        "metrics": parse_result(result_file.read_text()),
        "result_file": str(result_file),
        "evaluator_log": str(log),
        "evaluator_runtime_s": round(time.perf_counter() - start, 1),
        "predictions": stats,
    }
    (pred_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def benchmark(
    pairs: Sequence[str], out_root: str | Path, cfg: Mapping[str, Any], skip_run: bool = False,
    config_file: str | Path | None = None, overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """`roomdiff eval`: run (unless `skip_run`) the pipeline on every pair, write the
    predictions, evaluate them and print the main numbers. Returns the summary."""
    out_root = Path(out_root).resolve()
    run_pairs(pairs, out_root, cfg, config_file, overrides, skip_run)
    summary = evaluate(out_root / "pred", pairs, root=data_root(cfg))
    m = summary["metrics"]
    print(f"[eval] {m['scenes']} scenes: per-view AP {m['per_view_ap']:.4f}, per-scene AP "
          f"{m['per_scene_ap']:.4f}, AP_type {m['per_scene_ap_type']:.4f}, global IoU "
          f"{m['global_iou']:.4f} -> {out_root / 'pred' / 'summary.json'}")
    for line in summary["evaluator_errors"]:
        print(f"[eval] evaluator: {line}")
    return summary
