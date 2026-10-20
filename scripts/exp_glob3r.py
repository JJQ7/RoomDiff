"""Glob3R experiment harness: swap the Pi3X poses and depth of finished SceneDiff runs for
Glob3R's and measure the alignment and the benchmark AP on the same pairs.

Usage: python scripts/exp_glob3r.py <command> [options]   (from the repository root)

  prepare     <dst>/runs/<pair>/: a copy of the baseline keyframes/ (what Glob3R reads) and
              recon_pi3x -> the baseline recon/; <base>/runs/<pair> -> the baseline run.
  convert     <dst>/runs/<pair>/glob3r/glob3r_raw.npz -> <dst>/runs/<pair>/recon/ in the M2
              contract (docs/interfaces.md), levelled and diagnosed like `run_recon`, and by
              default brought onto the Pi3X metric scale (one global factor, the median
              per-frame Glob3R / Pi3X depth ratio; Glob3R's own frame is normalised).
              --from-baseline: round trip of the Pi3X recon through the same contract at
              another resolution (writes <dst>/validate/<pair>/recon/), the converter test.
              --identity: a verbatim copy of the Pi3X recon (re-run noise control).
  compare     alignment diagnostics baseline vs converted -> <dst>/compare_alignment.md/.json
  downstream  visibility .. describe on the converted runs, one guarded GPU job per stage;
              writes <run>/eval_run.json so that `roomdiff eval --skip-run` sees done / failed.
  score       the official evaluator on <base> and <dst> over the same pairs ->
              <dst>/compare_ap.md/.json (--base-only: the baseline alone).

Raw file contract (`glob3r_raw.npz`, frames in keyframes/index.json order, before first):
files (N,) str, rec (N,) uint8, cam2world (N,4,4), K (N,3,3) at the depth resolution,
depth (N,H',W') z-depth (0 / NaN unknown), conf (N,H',W') in [0,1], has_depth (N,) bool,
scale (N,) float; `glob3r_meta.json` next to it (mode, infer_args, n_frames, n_with_depth,
resolution [W, H], runtime_s, peak_vram_gb, glob3r_commit, notes, optional metric_scale).

Resampling convention: pixel centres at integer coordinates (as `unproject_depth` and Pi3's
`recover_intrinsic_from_rays_d`). Depth / conf are resized by centre-aligned nearest
neighbour and K by f' = s f, c' = (c + 0.5) s - 0.5 with s = new / old size, so a depth
pixel and its K-ray stay consistent (plain c' = s c differs by 0.5 (s - 1) px).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # roomdiff without installing
from roomdiff.config import load_config  # noqa: E402
from roomdiff.eval_scenediff import RUN_MARKER, run_state  # noqa: E402
from roomdiff.geometry import sim3_apply_to_poses, unproject_depth  # noqa: E402
from roomdiff.paths import RECORDINGS  # noqa: E402
from roomdiff.recon import (  # noqa: E402
    alignment_diagnostics,
    level_transform,
    load_recon,
    verdict_line,
    write_ply,
    write_recon,
)

DEFAULT_PAIRS = ROOT / "outputs" / "exp_glob3r" / "pairs_fairweak.txt"
DEFAULT_SRC = ROOT / "outputs" / "scenediff_eval_val_full"
DEFAULT_DST = ROOT / "outputs" / "exp_glob3r"
DEFAULT_BASE = ROOT / "outputs" / "exp_glob3r_base"
GUARD = ROOT / "scripts" / "run_guarded.sh"
PYTHON = ROOT / ".venv" / "bin" / "python"
GUARD_ARGS = ("-g", "-m", "10G", "-t", "1500")
INGEST_OVERRIDES = ("ingest.mode=fixed", "ingest.fixed_interval_s=0.5")
STAGES = ("visibility", "appearance", "instances", "changes", "describe")
# outputs a stage (re)writes: removed before it runs again (never keyframes/, recon/, ...)
STAGE_OUTPUTS = {
    "visibility": ("visibility",),
    "appearance": ("appearance",),
    "instances": ("instances",),
    "changes": ("changes.json", "crops"),
    "describe": ("semantics", "report.md"),
}
ALWAYS_STALE = ("viewer", RUN_MARKER)  # plus every *.rrd in the run directory
PROTECTED = ("keyframes", "recon", "recon_pi3x", "glob3r")
LOW_MEMORY_RC, LOW_MEMORY_WAIT_S = 75, 60  # run_guarded.sh: < 6 GB free, try again later
RAW_KEYS = ("files", "rec", "cam2world", "K", "depth", "conf", "has_depth", "scale")
MIN_OVERLAP_PIXELS = 100  # scale ratio: frames with fewer pixels valid in both are skipped
MAX_ASPECT_DIFF = 0.02  # raw vs baseline aspect ratio (a crop would misplace every pixel)
SCALE_WARN = 0.05  # warn if Glob3R's scale differs from Pi3X's by more than this
POSE_TOL = 1e-3  # max deviation of a cam2world rotation from orthonormal
EDGE_CHUNK = 8  # frames per depth_normal_edge call (bounded CPU memory)
VERDICT_RANK = {"alignment WEAK": 0, "alignment FAIR": 1, "alignment OK": 2}
AP_METRICS = ("per_view_ap", "per_scene_ap", "per_scene_ap_type", "global_iou")
COUNT_METRICS = ("scenes", "object_tp", "object_fp", "object_fn")

NA = "-"  # table cell without a value
EdgeFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


# ------------------------------------------------------------------ helpers


def read_pairs(path: str | Path) -> list[str]:
    """Pair names, one per line (blank lines and # comments skipped)."""
    lines = (line.split("#", 1)[0].strip() for line in Path(path).read_text().splitlines())
    return [line for line in lines if line]


def run_dir(root: Path, pair: str) -> Path:
    return Path(root) / "runs" / pair


def short_verdict(verdict: str | None) -> str:
    return (verdict or "n/a").replace("alignment ", "")


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def resize_nearest(array: np.ndarray, height: int, width: int) -> np.ndarray:
    """(N, h, w, ...) array resized to (N, height, width, ...) by centre-aligned nearest
    neighbour: output pixel u takes input pixel floor((u + 0.5) w / width). No values are
    mixed, so depth edges stay sharp."""
    h, w = array.shape[1:3]
    rows = np.minimum(((np.arange(height) + 0.5) * h / height).astype(int), h - 1)
    cols = np.minimum(((np.arange(width) + 0.5) * w / width).astype(int), w - 1)
    return array[:, rows][:, :, cols]


def rescale_intrinsics(
    K: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]
) -> np.ndarray:
    """Intrinsics (N, 3, 3) of images (src H, W) for the same images resized to (dst H, W):
    fx (and skew) by W'/W, fy by H'/H, principal point c' = (c + 0.5) s - 0.5 (pixel
    centres at integer coordinates, matching `resize_nearest`)."""
    sy, sx = dst_hw[0] / src_hw[0], dst_hw[1] / src_hw[1]
    out = np.array(K, np.float64)
    out[:, 0, :2] *= sx
    out[:, 1, 1] *= sy
    out[:, 0, 2] = (out[:, 0, 2] + 0.5) * sx - 0.5
    out[:, 1, 2] = (out[:, 1, 2] + 0.5) * sy - 0.5
    return out.astype(np.float32)


def check_frames(raw: Mapping[str, Any], base: Mapping[str, Any]) -> None:
    """Raise ValueError unless the raw file holds the baseline's frames in its order
    (file names compared without any directory part) with the same before / after split."""
    raw_files = [Path(str(f)).name for f in np.asarray(raw["files"]).tolist()]
    base_files = [str(f) for f in np.asarray(base["files"]).tolist()]
    if len(raw_files) != len(base_files):
        raise ValueError(f"raw file has {len(raw_files)} frames, the baseline recon "
                         f"{len(base_files)}: Glob3R must return every keyframe")
    wrong = [i for i, (a, b) in enumerate(zip(raw_files, base_files)) if a != b]
    if wrong:
        i = wrong[0]
        raise ValueError(f"raw `files` differ from the baseline at {len(wrong)} positions, "
                         f"first at {i}: {raw_files[i]!r} vs {base_files[i]!r} (frames must "
                         "be in keyframes/index.json order, all before then all after)")
    raw_rec = np.asarray(raw["rec"]).astype(np.int64)
    if not np.array_equal(raw_rec, np.asarray(base["rec"]).astype(np.int64)):
        bad = int(np.flatnonzero(raw_rec != np.asarray(base["rec"]))[0])
        raise ValueError(f"raw `rec` differs from the baseline (first at frame {bad}, "
                         f"{raw_files[bad]}): 0 = before, 1 = after")


def check_raw(raw: Mapping[str, Any]) -> None:
    """Raise ValueError if keys or shapes break the raw contract."""
    missing = [k for k in RAW_KEYS if k not in raw]
    if missing:
        raise ValueError(f"raw file lacks {missing} (contract keys: {list(RAW_KEYS)})")
    n = len(raw["files"])
    depth = np.asarray(raw["depth"])
    expected = {"rec": (n,), "cam2world": (n, 4, 4), "K": (n, 3, 3), "has_depth": (n,),
                "scale": (n,), "conf": depth.shape}
    for key, shape in expected.items():
        if np.shape(raw[key]) != shape:
            raise ValueError(f"raw[{key!r}] has shape {np.shape(raw[key])}, expected {shape}")
    if depth.ndim != 3 or len(depth) != n:
        raise ValueError(f"raw['depth'] has shape {depth.shape}, expected ({n}, H', W')")
    poses = np.asarray(raw["cam2world"])
    bad = np.flatnonzero(~np.isfinite(poses).all(axis=(1, 2)))
    if len(bad):
        names = [str(raw["files"][i]) for i in bad[:5]]
        raise ValueError(f"{len(bad)} frames have a non-finite cam2world (first: {names}); "
                         "every keyframe needs a pose")
    R = poses[:, :3, :3].astype(np.float64)
    error = np.abs(R.transpose(0, 2, 1) @ R - np.eye(3)).max(axis=(1, 2))
    error = np.maximum(error, np.abs(np.linalg.det(R) - 1))
    error = np.maximum(error, np.abs(poses[:, 3] - [0, 0, 0, 1]).max(axis=1))
    if (error > POSE_TOL).any():
        i = int(np.argmax(error))
        raise ValueError(f"cam2world of {raw['files'][i]} is not a rigid transform (rotation "
                         f"error {error[i]:.2e}); poses must be camera-to-world SE(3)")


def load_raw(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """`glob3r_raw.npz` and the `glob3r_meta.json` next to it ({} if missing)."""
    try:
        with np.load(path, allow_pickle=False) as f:
            raw = {key: f[key] for key in f.files}
    except ValueError as err:  # object arrays need pickle
        raise ValueError(f"{path}: {err}; store `files` as a unicode array "
                         "(np.array(names)), not dtype=object") from None
    meta = load_json(path.with_name("glob3r_meta.json")) or {}
    return raw, meta


# ------------------------------------------------------------------ conversion


def pi3_edges(local: np.ndarray, mask: np.ndarray, rtol: float) -> np.ndarray:
    """`pi3.utils.geometry.depth_normal_edge` (as in `run_pi3x`) on CPU, in chunks."""
    import torch
    from pi3.utils.geometry import depth_normal_edge

    out = np.zeros(mask.shape, bool)
    for i in range(0, len(local), EDGE_CHUNK):
        with torch.no_grad():
            points = torch.from_numpy(np.ascontiguousarray(local[i:i + EDGE_CHUNK]))
            edge = depth_normal_edge(points, rtol=rtol,
                                     mask=torch.from_numpy(mask[i:i + EDGE_CHUNK]))
        out[i:i + EDGE_CHUNK] = edge.numpy()
    return out


def valid_mask(
    depth: np.ndarray, K: np.ndarray, conf: np.ndarray, conf_threshold: float,
    edge_rtol: float, edge_fn: EdgeFn | None = None,
) -> np.ndarray:
    """The `run_pi3x` rule: conf > threshold and not a depth/normal edge of the local point
    map (here unprojected from `depth` with `K`); also depth finite and > 0."""
    edge_fn = edge_fn or pi3_edges
    with np.errstate(invalid="ignore"):
        good = np.isfinite(depth) & (depth > 0) & (np.nan_to_num(conf) > conf_threshold)
    clean = np.where(good, depth, 0.0).astype(np.float32)
    local = np.stack([unproject_depth(clean[i], K[i], np.eye(4)) for i in range(len(depth))])
    return good & ~edge_fn(local, good, edge_rtol)


def scale_ratio(
    raw_depth: np.ndarray, base_depth: np.ndarray, base_valid: np.ndarray,
    frames: Sequence[int],
) -> tuple[float | None, int]:
    """Median over `frames` of median(raw depth) / median(baseline depth) on the pixels
    valid in both (raw depth finite > 0, baseline valid); frames with fewer than
    MIN_OVERLAP_PIXELS such pixels are skipped. Returns (ratio or None, frames used)."""
    ratios = []
    for i in frames:
        with np.errstate(invalid="ignore"):
            both = np.isfinite(raw_depth[i]) & (raw_depth[i] > 0) & base_valid[i] & \
                (base_depth[i] > 0)
        if both.sum() >= MIN_OVERLAP_PIXELS:
            ratios.append(float(np.median(raw_depth[i][both]) / np.median(base_depth[i][both])))
    return (float(np.median(ratios)) if ratios else None), len(ratios)


def fill_fallback(
    depth: np.ndarray, conf: np.ndarray, has_depth: np.ndarray, base: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, float | None, int]:
    """Depth and conf at the baseline resolution with the frames without Glob3R depth
    replaced by the baseline Pi3X depth times the scale ratio (z-depth does not depend on
    the pose) and the baseline conf. Returns (depth, conf, ratio, frames used for it)."""
    has_depth = np.asarray(has_depth, bool)
    ratio, used = scale_ratio(depth, base["depth"], base["valid"], np.flatnonzero(has_depth))
    missing = np.flatnonzero(~has_depth)
    depth, conf = depth.copy(), conf.copy()
    if len(missing):
        if ratio is None:
            raise ValueError(f"{len(missing)} frames have no Glob3R depth and no frame with "
                             "depth overlaps the baseline enough to transfer its scale")
        depth[missing] = base["depth"][missing].astype(np.float32) * ratio
        conf[missing] = base["conf"][missing].astype(np.float32)
    return depth, conf, ratio, used


def synthesise_raw(
    base: Mapping[str, Any], factor: float = 0.75, drop_every: int = 0
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A raw dict made from the baseline recon itself (the converter's round-trip test):
    the levelled Pi3X poses, depth and conf resized by `factor` (nearest neighbour, K to
    match), has_depth True except every `drop_every`-th frame (0: none)."""
    n, h, w = base["depth"].shape
    size = (max(1, round(h * factor)), max(1, round(w * factor)))
    has_depth = np.ones(n, bool)
    if drop_every > 0:
        has_depth[drop_every - 1::drop_every] = False
    depth = resize_nearest(base["depth"].astype(np.float32), *size)
    depth[~has_depth] = 0.0
    raw = {"files": np.asarray(base["files"]).copy(), "rec": np.asarray(base["rec"]).copy(),
           "cam2world": base["cam2world"].astype(np.float32),
           "K": rescale_intrinsics(base["K"], (h, w), size), "depth": depth,
           "conf": resize_nearest(base["conf"].astype(np.float32), *size),
           "has_depth": has_depth, "scale": np.ones(n, np.float32)}
    meta = {"mode": "from-baseline", "infer_args": {"factor": factor, "drop_every": drop_every},
            "n_frames": n, "n_with_depth": int(has_depth.sum()),
            "resolution": [size[1], size[0]], "runtime_s": 0.0, "peak_vram_gb": None,
            "glob3r_commit": None, "metric_scale": base["metric_scale"],
            "notes": "identity round trip of the Pi3X recon through the glob3r_raw contract"}
    return raw, meta


def convert(
    raw: Mapping[str, Any], raw_meta: Mapping[str, Any], base: Mapping[str, Any],
    base_meta: Mapping[str, Any], params: Mapping[str, Any], out: Path,
    edge_fn: EdgeFn | None = None, match_pi3x_scale: bool = True,
) -> dict[str, Any]:
    """Write `<out>/recon.npz`, `meta.json`, `before.ply`, `after.ply` (M2 contract) from a
    raw Glob3R result and the baseline recon of the same keyframes. `match_pi3x_scale`:
    divide translations and depth by `scale_ratio_vs_pi3x` (Glob3R works in a normalised
    frame, not metres; downstream thresholds are metric). Returns meta.json."""
    start = time.perf_counter()
    times: dict[str, float] = {}
    check_raw(raw)
    check_frames(raw, base)
    _, h, w = base["depth"].shape
    raw_h, raw_w = np.shape(raw["depth"])[1:]
    aspect = (raw_w / raw_h) / (w / h)
    if abs(aspect - 1) > MAX_ASPECT_DIFF:
        raise ValueError(f"raw depth is {raw_w}x{raw_h}, the baseline {w}x{h}: aspect ratios "
                         f"differ by {100 * abs(aspect - 1):.1f}% (cropped?)")

    has_depth = np.asarray(raw["has_depth"], bool)
    depth = resize_nearest(np.asarray(raw["depth"], np.float32), h, w)
    K = rescale_intrinsics(raw["K"], (raw_h, raw_w), (h, w))
    raw_conf = np.asarray(raw["conf"], np.float32)
    conf_all_ones = bool(np.all(raw_conf[has_depth] == 1.0)) if has_depth.any() else True
    if conf_all_ones:
        conf, conf_source = base["conf"].astype(np.float32), "pi3x (glob3r conf all ones)"
    else:
        conf, conf_source = resize_nearest(np.nan_to_num(raw_conf), h, w), "glob3r"
    depth, conf, ratio, ratio_frames = fill_fallback(depth, conf, has_depth, base)
    if not conf_all_ones and (~has_depth).any():
        conf_source += " (pi3x on the fallback frames)"
    cam2world = np.asarray(raw["cam2world"], np.float32).copy()
    if match_pi3x_scale and ratio is not None:  # back onto the Pi3X metric scale
        cam2world[:, :3, 3] /= ratio
        depth = depth / ratio
    with np.errstate(invalid="ignore"):
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
    valid = valid_mask(depth, K, conf, params["conf_threshold"], params["edge_rtol"], edge_fn)
    times["resample_and_valid"] = time.perf_counter() - start

    t0 = time.perf_counter()
    rec = np.asarray(base["rec"], np.uint8)
    pred = {"cam2world": cam2world, "K": K, "depth": depth, "conf": conf, "valid": valid}
    T, info = level_transform({**pred, "rec": rec}, params["snap_to_walls"],
                              params["wall_peak_fraction"])
    pred["cam2world"], _ = sim3_apply_to_poses(T, pred["cam2world"])
    level = {"levelled": True, "up_axis_original": info.pop("up_axis_original"),
             "floor_inliers_fraction": info.pop("floor_inliers_fraction"),
             **{k: info.pop(k) for k in ("walls_snapped", "wall_yaw_deg",
                                         "wall_peak_fraction") if k in info},
             "level": info}
    times["level"] = time.perf_counter() - t0
    scale = params["scale_factor"]
    pred["cam2world"][:, :3, 3] *= scale
    pred["depth"] = pred["depth"] * scale
    metric = float(raw_meta.get("metric_scale") or 1.0)
    recon = {"rec": rec, "files": np.asarray(base["files"]), "images": base["images"],
             **pred, "metric_scale": metric}
    out.mkdir(parents=True, exist_ok=True)
    write_recon(out / "recon.npz", recon)

    t0 = time.perf_counter()
    diag, clouds = alignment_diagnostics(recon, params["voxel_size_diag"])
    for name, (points, colors) in zip(RECORDINGS, clouds):
        write_ply(out / f"{name}.ply", points, colors)
    times["diagnostics"] = time.perf_counter() - t0
    times["total"] = time.perf_counter() - start

    scales = np.asarray(raw["scale"], np.float64)
    meta = {
        "model": "glob3r",
        "mode": raw_meta.get("mode"),
        "resolution": [w, h],
        "frames": {name: int((rec == r).sum()) for r, name in enumerate(RECORDINGS)},
        "metric_scale": metric,
        "scale_factor": scale,
        "params": dict(params),
        "runtime_s": {"glob3r": raw_meta.get("runtime_s"),
                      **{k: round(v, 2) for k, v in times.items()}},
        "peak_vram_gb": raw_meta.get("peak_vram_gb"),
        "alignment": {"method": "glob3r"},
        **level,
        "diagnostics": diag,
        "source": "glob3r",
        "raw_meta": dict(raw_meta),
        "raw_resolution": [raw_w, raw_h],
        "baseline_diagnostics": base_meta.get("diagnostics"),
        "fallback_frames": int((~has_depth).sum()),
        "scale_ratio_vs_pi3x": ratio,
        "scale_ratio_frames": ratio_frames,
        "matched_pi3x_scale": bool(match_pi3x_scale and ratio is not None),
        "conf_source": conf_source,
        "glob3r_frame_scale": {"min": float(scales.min()), "median": float(np.median(scales)),
                               "max": float(scales.max())},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def convert_pair(args: argparse.Namespace, pair: str, edge_fn: EdgeFn | None = None) -> None:
    """`convert` of one pair (also --from-baseline / --identity); prints both verdicts."""
    base_dir = run_dir(args.src, pair) / "recon"
    base_meta = json.loads((base_dir / "meta.json").read_text())
    if args.out:
        out = Path(args.out)
    elif args.from_baseline:
        out = Path(args.dst) / "validate" / pair / "recon"
    else:
        out = run_dir(args.dst, pair) / "recon"
    if args.identity:
        out.mkdir(parents=True, exist_ok=True)
        for name in ("recon.npz", "meta.json", "before.ply", "after.ply"):
            shutil.copy2(base_dir / name, out / name)
        meta = json.loads((out / "meta.json").read_text())
        meta.update(source="pi3x-identity", baseline_diagnostics=meta["diagnostics"])
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[convert] {pair}: Pi3X recon copied -> {out}")
        if out == run_dir(args.dst, pair) / "recon":
            (out.parent / RUN_MARKER).unlink(missing_ok=True)
        return
    base = load_recon(base_dir / "recon.npz")
    if args.from_baseline:
        raw, raw_meta = synthesise_raw(base, args.factor, args.drop_every)
    else:
        raw_path = Path(args.raw) if args.raw else \
            run_dir(args.dst, pair) / "glob3r" / "glob3r_raw.npz"
        if not raw_path.is_file():
            raise SystemExit(f"[convert] {pair}: no raw file {raw_path} (Glob3R output not "
                             "delivered yet?)")
        raw, raw_meta = load_raw(raw_path)
    params = load_config()["recon"]
    meta = convert(raw, raw_meta, base, base_meta, params, out, edge_fn, args.match_pi3x_scale)
    marker = out.parent / RUN_MARKER
    if out == run_dir(args.dst, pair) / "recon" and marker.is_file():
        marker.unlink()  # a new recon: the downstream run is stale (`downstream` re-runs it)
        print(f"[convert] {pair}: removed the stale {marker.name}; run `downstream` again")
    ratio = meta["scale_ratio_vs_pi3x"]
    print(f"[convert] {pair}: {meta['fallback_frames']} fallback frames, scale vs Pi3X "
          + (f"{ratio:.4f}" if ratio is not None else "n/a")
          + f", conf {meta['conf_source']}, {meta['runtime_s']['total']:.1f} s -> {out}")
    if ratio is not None and abs(ratio - 1) > SCALE_WARN and not meta["matched_pi3x_scale"]:
        print(f"[convert] WARNING {pair}: Glob3R depth is {ratio:.3f} x Pi3X's; downstream "
              "thresholds are metric (re-run without --keep-glob3r-scale)")
    old, new = base_meta["diagnostics"], meta["diagnostics"]
    print(f"[convert] baseline: {verdict_line(old)}")
    print(f"[convert] glob3r:   {verdict_line(new)}")
    a, b = old.get("frac_lt_3cm_min"), new.get("frac_lt_3cm_min")
    if a is not None and b is not None:
        print(f"[convert] min <3cm {a:.4f} -> {b:.4f} ({b - a:+.4f})")


# ------------------------------------------------------------------ prepare


def link(path: Path, target: Path) -> None:
    """Symlink `path` -> absolute `target`, replacing an older link (never a real dir)."""
    target = target.absolute()
    if path.is_symlink():
        if Path(path.readlink()) == target:
            return
        path.unlink()
    elif path.exists():
        raise FileExistsError(f"{path} exists and is not a symlink; remove it by hand")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=True)


def prepare(args: argparse.Namespace) -> None:
    for pair in read_pairs(args.pairs):
        src = run_dir(args.src, pair).absolute()
        if not (src / "keyframes" / "index.json").is_file() or \
                not (src / "recon" / "recon.npz").is_file():
            raise FileNotFoundError(f"{src}: no keyframes/index.json or recon/recon.npz")
        run = run_dir(args.dst, pair)
        run.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src / "keyframes", run / "keyframes", dirs_exist_ok=True)
        link(run / "recon_pi3x", src / "recon")
        link(run_dir(args.base, pair), src)
        print(f"[prepare] {pair}: {run} (keyframes copied, recon_pi3x linked); "
              f"{run_dir(args.base, pair)} -> {src}")


# ------------------------------------------------------------------ compare


def _get(d: Mapping[str, Any] | None, *keys: str) -> Any:
    for key in keys:
        if not isinstance(d, Mapping) or key not in d:
            return None
        d = d[key]
    return d


def alignment_row(pair: str, base: Mapping[str, Any], new: Mapping[str, Any] | None
                  ) -> dict[str, Any]:
    """Baseline vs converted numbers of one pair (new = None: not converted)."""
    row: dict[str, Any] = {"pair": pair}
    for side, meta in (("base", base), ("new", new)):
        diag = (meta or {}).get("diagnostics")
        row[side] = None if diag is None else {
            "verdict": diag.get("verdict"),
            "frac_lt_3cm_min": diag.get("frac_lt_3cm_min"),
            **{f"covis_{k}_{d}": _get(diag, "nn_covisible", f"{a}_to_{b}", f)
               for k, f in (("median", "median_m"), ("fraction", "fraction"))
               for d, (a, b) in (("ba", RECORDINGS), ("ab", RECORDINGS[::-1]))},
            **{f"all_median_{d}": _get(diag, "nn_all", f"{a}_to_{b}", "median_m")
               for d, (a, b) in (("ba", RECORDINGS), ("ab", RECORDINGS[::-1]))},
            "valid_fraction": [_get(diag, r, "valid_fraction") for r in RECORDINGS],
        }
    row["fallback_frames"] = (new or {}).get("fallback_frames")
    row["scale_ratio_vs_pi3x"] = (new or {}).get("scale_ratio_vs_pi3x")
    old_f, new_f = _get(row, "base", "frac_lt_3cm_min"), _get(row, "new", "frac_lt_3cm_min")
    row["delta_frac_lt_3cm_min"] = None if old_f is None or new_f is None else new_f - old_f
    if row["new"] is not None:
        old_v = VERDICT_RANK.get(row["base"]["verdict"], -1)
        new_v = VERDICT_RANK.get(row["new"]["verdict"], -1)
        row["verdict_change"] = "improved" if new_v > old_v else \
            "worsened" if new_v < old_v else "unchanged"
    return row


def _fmt(value: Any, scale: float = 1.0, digits: int = 3) -> str:
    return NA if value is None else f"{scale * value:.{digits}f}"


def _signed(value: float | None, digits: int = 3) -> str:
    return NA if value is None else f"{value:+.{digits}f}"


def _pair_cell(row: Mapping[str, Any], key: str, scale: float = 1.0, digits: int = 3) -> str:
    old, new = _get(row, "base", key), _get(row, "new", key)
    return _fmt(old, scale, digits) if row["new"] is None else \
        f"{_fmt(old, scale, digits)} → {_fmt(new, scale, digits)}"


def alignment_table(rows: Sequence[Mapping[str, Any]]) -> str:
    head = ("| pair | verdict | min <3cm | Δ | covis median b→a cm | covis median a→b cm | "
            "coverage b→a | coverage a→b | all median b→a cm | all median a→b cm | "
            "valid b / a | fallback | s vs Pi3X |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for row in rows:
        verdict = short_verdict(_get(row, "base", "verdict"))
        if row["new"] is not None:
            verdict += f" → {short_verdict(row['new']['verdict'])}"
        side = row["new"] or row["base"]
        valid = " / ".join(_fmt(v, 1, 2) for v in side["valid_fraction"])
        lines.append(
            f"| {row['pair']} | {verdict} | {_pair_cell(row, 'frac_lt_3cm_min')} | "
            f"{_signed(row['delta_frac_lt_3cm_min'])} | "
            f"{_pair_cell(row, 'covis_median_ba', 100, 2)} | "
            f"{_pair_cell(row, 'covis_median_ab', 100, 2)} | "
            f"{_pair_cell(row, 'covis_fraction_ba', 1, 2)} | "
            f"{_pair_cell(row, 'covis_fraction_ab', 1, 2)} | "
            f"{_pair_cell(row, 'all_median_ba', 100, 2)} | "
            f"{_pair_cell(row, 'all_median_ab', 100, 2)} | {valid} | "
            f"{_fmt(row['fallback_frames'], 1, 0)} | {_fmt(row['scale_ratio_vs_pi3x'], 1, 4)} |")
    return "\n".join(lines)


def alignment_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    done = [r for r in rows if r["new"] is not None]
    deltas = [r["delta_frac_lt_3cm_min"] for r in done if r["delta_frac_lt_3cm_min"] is not None]
    counts = {k: sum(r["verdict_change"] == k for r in done)
              for k in ("improved", "unchanged", "worsened")}
    return {"pairs": len(rows), "converted": len(done), **counts,
            "mean_delta_frac_lt_3cm_min": float(np.mean(deltas)) if deltas else None,
            "median_delta_frac_lt_3cm_min": float(np.median(deltas)) if deltas else None,
            "verdicts_new": {v: sum(_get(r, "new", "verdict") == v for r in done)
                             for v in VERDICT_RANK}}


def compare(args: argparse.Namespace) -> None:
    rows = []
    for pair in read_pairs(args.pairs):
        base = json.loads((run_dir(args.src, pair) / "recon" / "meta.json").read_text())
        new_dir = (Path(args.dst) / "validate" / pair if args.validate
                   else run_dir(args.dst, pair)) / "recon"
        rows.append(alignment_row(pair, base, load_json(new_dir / "meta.json")))
    summary = alignment_summary(rows)
    table = alignment_table(rows)
    mean = summary["mean_delta_frac_lt_3cm_min"]
    text = (f"# Alignment: Pi3X baseline → {'round trip' if args.validate else 'Glob3R'}\n\n"
            f"{summary['converted']} of {summary['pairs']} pairs converted; verdict improved "
            f"{summary['improved']}, unchanged {summary['unchanged']}, worsened "
            f"{summary['worsened']}; mean Δ min <3cm "
            + ("n/a" if mean is None else f"{mean:+.4f}") + ".\n\n"
            "Cells: baseline → new. min <3cm = `frac_lt_3cm_min` (co-visible, weaker "
            "direction); coverage = co-visible `fraction`; valid = valid pixel share per "
            "recording (new); s = median Glob3R / Pi3X depth.\n\n" + table + "\n")
    name = "compare_alignment_validate" if args.validate else "compare_alignment"
    Path(args.dst).mkdir(parents=True, exist_ok=True)
    (Path(args.dst) / f"{name}.md").write_text(text)
    (Path(args.dst) / f"{name}.json").write_text(
        json.dumps({"summary": summary, "pairs": rows}, indent=2))
    print(text)


# ------------------------------------------------------------------ downstream


def stage_command(stage: str, run: Path, extra_sets: Sequence[str] = ()) -> list[str]:
    cmd = [str(GUARD), *GUARD_ARGS, "--", str(PYTHON), "-m", "roomdiff.cli", stage,
           "--out", str(run)]
    for item in (*INGEST_OVERRIDES, *extra_sets):
        cmd += ["--set", item]
    return cmd


def remove_stale(run: Path, stages: Sequence[str]) -> list[str]:
    """Delete the outputs of `stages` and of every later stage, the viewer, *.rrd and the
    run marker in `run`; keyframes/, recon/, recon_pi3x, glob3r/ are never touched.
    Returns the names removed."""
    first = min(STAGES.index(s) for s in stages)
    names = [n for s in STAGES[first:] for n in STAGE_OUTPUTS[s]]
    names += [*ALWAYS_STALE, *(p.name for p in sorted(run.glob("*.rrd")))]
    removed = []
    for name in names:
        if name in PROTECTED or "/" in name:
            raise ValueError(f"refusing to remove {name}")
        path = run / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            continue
        removed.append(name)
    return removed


def run_logged(cmd: Sequence[str], log: Path, retries: int = 10,
               wait_s: float = LOW_MEMORY_WAIT_S) -> int:
    """Run `cmd` from the repository root, appending to `log`; on run_guarded.sh's
    low-memory refusal (75) wait and retry up to `retries` times. Returns the exit code."""
    for attempt in range(retries + 1):
        with open(log, "a") as f:
            f.write(f"\n$ {' '.join(cmd)}\n")
            f.flush()
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                check=False).returncode
        if rc != LOW_MEMORY_RC or attempt == retries:
            return rc
        print(f"[downstream] less than 6 GB RAM free, retrying in {wait_s:.0f} s")
        time.sleep(wait_s)
    return rc


def downstream_pair(
    run: Path, log: Path, stages: Sequence[str], extra_sets: Sequence[str] = (),
    runner: Callable[[Sequence[str], Path], int] = run_logged,
) -> dict[str, Any]:
    """Run `stages` on one converted run, stop at the first failure, write eval_run.json."""
    remove_stale(run, stages)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"[downstream] {run}\n")
    start, rc, cmds, failed = time.perf_counter(), 0, [], None
    for stage in stages:
        cmd = stage_command(stage, run, extra_sets)
        cmds.append(cmd)
        t0 = time.perf_counter()
        rc = runner(cmd, log)
        print(f"[downstream] {run.name}: {stage} rc {rc} in {time.perf_counter() - t0:.0f} s")
        if rc != 0:
            failed = stage
            break
    marker = {"rc": rc, "seconds": round(time.perf_counter() - start, 1), "cmd": cmds,
              "stages": list(stages), "failed_stage": failed}
    (run / RUN_MARKER).write_text(json.dumps(marker, indent=1))
    return marker


def parse_stages(text: str) -> list[str]:
    stages = [s.strip() for s in text.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown or not stages:
        raise argparse.ArgumentTypeError(f"unknown stage(s) {unknown}; choose from {STAGES}")
    return stages


def downstream(args: argparse.Namespace) -> None:
    pairs = read_pairs(args.pairs)
    for i, pair in enumerate(pairs, start=1):
        run = run_dir(args.dst, pair)
        if not (run / "recon" / "recon.npz").is_file():
            print(f"[downstream] {i}/{len(pairs)} {pair}: no converted recon, skipped")
            continue
        marker = load_json(run / RUN_MARKER)
        if marker and marker["rc"] == 0 and not args.force:
            print(f"[downstream] {i}/{len(pairs)} {pair}: done before (--force to re-run)")
            continue
        print(f"[downstream] {i}/{len(pairs)} {pair}: {', '.join(args.stages)}")
        marker = downstream_pair(run, Path(args.dst) / "logs" / f"{pair}.log", args.stages,
                                 args.set or [])
        print(f"[downstream] {pair}: rc {marker['rc']} in {marker['seconds']:.0f} s, "
              f"state {run_state(run)}")


# ------------------------------------------------------------------ score


def score_pairs(dst: Path, pairs: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    """Pairs with a converted recon and a finished downstream run in `dst`, and the reason
    each other pair is left out."""
    used, excluded = [], {}
    for pair in pairs:
        run = run_dir(dst, pair)
        marker = load_json(run / RUN_MARKER)
        if not (run / "recon" / "recon.npz").is_file():
            excluded[pair] = "no converted recon"
        elif marker is None:
            excluded[pair] = "no downstream run"
        elif marker["rc"] != 0 or run_state(run) != "done":
            excluded[pair] = f"downstream failed (rc {marker['rc']}, {marker.get('failed_stage')})"
        else:
            used.append(pair)
    return used, excluded


def evaluate_root(root: Path, pairs: Sequence[str], log: Path) -> dict[str, Any]:
    """`roomdiff eval --skip-run` on `root` for `pairs`; returns pred/summary.json."""
    summary_file = root / "pred" / "summary.json"
    summary_file.unlink(missing_ok=True)
    cmd = [str(PYTHON), "-m", "roomdiff.cli", "eval", "--pairs", ",".join(pairs),
           "--out-root", str(root), "--skip-run"]
    for item in INGEST_OVERRIDES:
        cmd += ["-o", item]
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[score] {root}: {len(pairs)} pairs (log {log})")
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                            check=False).returncode
    tail = log.read_text().splitlines()[-3:]
    print("\n".join(f"  {line}" for line in tail))
    if rc != 0 or not summary_file.is_file():
        raise RuntimeError(f"roomdiff eval on {root} failed (rc {rc}), see {log}")
    summary = json.loads(summary_file.read_text())
    if list(summary["pairs"]) != list(pairs):
        raise RuntimeError(f"{summary_file} lists other pairs than requested")
    return summary


def ap_table(results: Mapping[str, Mapping[str, Any]]) -> str:
    names = list(results)
    head = "| metric | " + " | ".join(names) + (" | Δ |" if len(names) == 2 else " |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for key in (*AP_METRICS, *COUNT_METRICS):
        values = [results[n]["metrics"].get(key) for n in names]
        if all(v is None for v in values):
            continue
        digits = 4 if key in AP_METRICS else 0
        cells = [_fmt(v, 1, digits) for v in values]
        if len(names) == 2:
            cells.append(_signed(None if None in values else values[1] - values[0], digits))
        label = {"object_tp": "found (object TP)", "object_fp": "object FP",
                 "object_fn": "missed (object FN)"}.get(key, key)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def score(args: argparse.Namespace) -> None:
    pairs = read_pairs(args.pairs)
    dst, base = Path(args.dst), Path(args.base)
    if args.base_only:
        used, excluded = list(pairs), {}
    else:
        used, excluded = score_pairs(dst, pairs)
    if not used:
        raise SystemExit("[score] no pair is ready to score")
    not_done = [p for p in used if run_state(run_dir(base, p).resolve()) != "done"]
    if not_done:
        print(f"[score] WARNING baseline runs not done (scored as such): {not_done}")
    roots = {"Pi3X (baseline)": base} if args.base_only else \
        {"Pi3X (baseline)": base, "Glob3R": dst}
    results = {}
    for name, root in roots.items():
        tag = "base" if root == base else "glob3r"
        summary = evaluate_root(root, used, dst / "logs" / f"score_{tag}.log")
        results[name] = {"root": str(root), "metrics": summary["metrics"],
                         "not_predicted": summary["not_predicted"],
                         "evaluator_errors": summary["evaluator_errors"],
                         "objects_predicted": {p: (s or {}).get("objects")
                                               for p, s in summary["predictions"].items()}}
    text = (f"# SceneDiff AP: {' vs '.join(results)}\n\n"
            f"{len(used)} pairs scored (official evaluator, `roomdiff eval --skip-run`)"
            + (f"; {len(excluded)} left out on both sides" if excluded else "") + ".\n\n"
            + ap_table(results) + "\n\n## Pairs scored\n\n"
            + "\n".join(f"- {p}" for p in used) + "\n")
    if excluded:
        text += "\n## Left out (both sides)\n\n" + "\n".join(
            f"- {p}: {why}" for p, why in excluded.items()) + "\n"
    errors = {n: r["evaluator_errors"] for n, r in results.items() if r["evaluator_errors"]}
    if errors:
        text += "\n## Evaluator errors\n\n" + "\n".join(
            f"- {n}: {e}" for n, es in errors.items() for e in es) + "\n"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "compare_ap.md").write_text(text)
    (dst / "compare_ap.json").write_text(json.dumps(
        {"pairs": used, "excluded": excluded, "results": results}, indent=2))
    print(text)


# ------------------------------------------------------------------ CLI


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS, help="pair list file")
        p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="baseline eval root")
        p.add_argument("--dst", type=Path, default=DEFAULT_DST, help="experiment root")
        p.add_argument("--base", type=Path, default=DEFAULT_BASE, help="baseline score root")
        return p

    command("prepare", "copy keyframes, link the baseline recon and runs")
    p = command("convert", "glob3r_raw.npz -> recon/ in the M2 contract")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--pair", action="append", help="pair name (repeatable)")
    which.add_argument("--all", action="store_true",
                       help="every pair of --pairs that has a raw file (or with --from-baseline)")
    p.add_argument("--raw", help="raw file (default <dst>/runs/<pair>/glob3r/glob3r_raw.npz)")
    p.add_argument("--out", help="recon directory to write (default <dst>/runs/<pair>/recon, "
                                 "with --from-baseline <dst>/validate/<pair>/recon)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--from-baseline", action="store_true",
                      help="round-trip the Pi3X recon through the raw contract (validation)")
    mode.add_argument("--identity", action="store_true",
                      help="copy the Pi3X recon verbatim (downstream re-run noise control)")
    p.add_argument("--factor", type=float, default=0.75,
                   help="--from-baseline: resolution factor of the synthetic raw file")
    p.add_argument("--drop-every", type=int, default=0,
                   help="--from-baseline: every n-th frame without depth (tests the fallback)")
    p.add_argument("--keep-glob3r-scale", dest="match_pi3x_scale", action="store_false",
                   help="keep Glob3R's units (default: divide poses and depth by the Glob3R / "
                        "Pi3X depth ratio, since downstream thresholds are metric)")
    command("compare", "alignment baseline vs converted").add_argument(
        "--validate", action="store_true", help="compare the --from-baseline round trips")
    p = command("downstream", "visibility .. describe on the converted runs")
    p.add_argument("--stages", type=parse_stages, default=list(STAGES),
                   help=f"comma-separated, in order (default {','.join(STAGES)})")
    p.add_argument("--set", action="append", help="extra config override for every stage")
    p.add_argument("--force", action="store_true", help="re-run pairs that finished")
    command("score", "official evaluator, baseline vs Glob3R").add_argument(
        "--base-only", action="store_true", help="score the baseline root only")
    args = parser.parse_args(argv)
    args.src, args.dst, args.base = (Path(p).absolute() for p in (args.src, args.dst, args.base))

    if args.command == "convert":
        if args.all:
            pairs = read_pairs(args.pairs)
            if not (args.from_baseline or args.identity):
                pairs = [p for p in pairs
                         if (run_dir(args.dst, p) / "glob3r" / "glob3r_raw.npz").is_file()]
            if args.raw or args.out:
                raise SystemExit("--raw / --out need a single --pair")
            if not pairs:
                print(f"[convert] no pair of {args.pairs} has a glob3r/glob3r_raw.npz yet")
        else:
            pairs = args.pair
            if len(pairs) > 1 and (args.raw or args.out):
                raise SystemExit("--raw / --out need a single --pair")
        for pair in pairs:
            convert_pair(args, pair)
    else:
        {"prepare": prepare, "compare": compare, "downstream": downstream,
         "score": score}[args.command](args)


if __name__ == "__main__":
    main()
