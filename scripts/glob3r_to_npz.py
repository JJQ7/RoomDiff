"""Convert one Glob3R output directory into glob3r_raw.npz + glob3r_meta.json (RoomDiff contract).

Usage (Glob3R venv: numpy + stdlib only):
  third_party/Glob3R/.venv/bin/python scripts/glob3r_to_npz.py <glob3r_dir> <index.json> <out_dir>
      [--run_info <run_info.json>]

<glob3r_dir> is infer.py's --save_path of a run with --save_frames (RoomDiff patch in
third_party/Glob3R, see glob3r/inference/frames_export.py); its input images were named
"<4-digit position>_<keyframe file>" by scripts/run_glob3r.sh, all before frames in index
order, then all after frames. Rows of frames.npz are in Glob3R's processing order (the
--unordered retrieval order) and carry their image path; this script maps them back to the
original keyframe order by file name and cross-checks the poses against poses_final.txt.

Output `glob3r_raw.npz` (N = all keyframes, original order):
  files (N,) str, rec (N,) uint8, cam2world (N,4,4) float32 (OpenCV axes, Glob3R units),
  K (N,3,3) float32 pinhole at the depth resolution, depth (N,H,W) float32 z-depth,
  conf (N,H,W) float32 in [0,1], has_depth (N,) bool, scale (N,) float32.
  Extras: is_keyframe (N,) bool, k1 (N,) float32, scale_points (N,) int64.
Glob3R's BA camera is SIMPLE_RADIAL (one focal, centre principal point, k1). depth and conf
are resampled (nearest) from the distorted network grid onto the undistorted pinhole grid of K,
so unprojecting depth with K reproduces Glob3R's own dense points; pixels that map outside
the network image get depth 0 and conf 0. K applies to integer pixel indices (u = column).
scale[i] = depth[i] / (raw Pi3X metric z-depth of the window that initialised frame i)
         = init_norm_scale * per-frame BA RANSAC scale; Glob3R units per metre ~ median(scale).
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def undistort_resample(depth, conf, K, k1):
    """Nearest-neighbour resample from the SIMPLE_RADIAL grid onto the pinhole grid of K."""
    h, w = depth.shape
    if k1 == 0.0:
        return depth, conf
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    xu = (u - K[0, 2]) / K[0, 0]
    yu = (v - K[1, 2]) / K[1, 1]
    f = 1.0 + k1 * (xu * xu + yu * yu)
    ud = np.rint(xu * f * K[0, 0] + K[0, 2]).astype(np.int64)
    vd = np.rint(yu * f * K[1, 1] + K[1, 2]).astype(np.int64)
    inside = (ud >= 0) & (ud < w) & (vd >= 0) & (vd < h)
    d_out = np.zeros_like(depth)
    c_out = np.zeros_like(conf)
    d_out[inside] = depth[vd[inside], ud[inside]]
    c_out[inside] = conf[vd[inside], ud[inside]]
    return d_out, c_out


def read_kitti(path):
    rows = np.loadtxt(path, ndmin=2).reshape(-1, 3, 4)
    out = np.tile(np.eye(4), (len(rows), 1, 1))
    out[:, :3, :] = rows
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("glob3r_dir")
    ap.add_argument("index_json")
    ap.add_argument("out_dir")
    ap.add_argument("--run_info", default=None, help="JSON written by run_glob3r.sh")
    ap.add_argument("--allow_crop", action="store_true",
                    help="convert even if the network input was cropped / padded "
                         "(scale_w != scale_h)")
    args = ap.parse_args(argv)
    t0 = time.time()

    with open(args.index_json) as f:
        index = json.load(f)
    files, rec = [], []
    for r, name in enumerate(("before", "after")):
        for fr in index[name]["frames"]:
            files.append(fr["file"])
            rec.append(r)
    flat = [f"{i:04d}_{f}" for i, f in enumerate(files)]
    n = len(files)

    fz = np.load(os.path.join(args.glob3r_dir, "frames.npz"))
    paths = [os.path.basename(str(p)) for p in fz["image_paths"]]
    if len(paths) != n or sorted(paths) != sorted(flat):
        sys.exit(f"frames.npz rows ({len(paths)}) do not match the {n} input files of index.json")
    row_of = np.array([paths.index(f) for f in flat])  # original position -> frames.npz row
    order = fz["order"]
    if not np.array_equal(order[row_of], np.arange(n)):
        sys.exit("frames.npz 'order' disagrees with its image_paths")

    w2c = fz["w2c"][row_of]
    c2w = np.linalg.inv(w2c)
    # poses_final.txt is camera-to-world in the sorted input order (= original order here)
    pf = read_kitti(os.path.join(args.glob3r_dir, "poses_final.txt"))
    pose_err = float(np.abs(pf - c2w).max())
    if pose_err > 1e-4 * max(1.0, float(np.abs(c2w[:, :3, 3]).max())):
        sys.exit("poses_final.txt disagrees with frames.npz after re-ordering "
                 f"(max err {pose_err})")

    sw, sh = (float(x) for x in fz["scale_wh"])
    if abs(sw / sh - 1.0) > 0.05 and not args.allow_crop:
        sys.exit(f"original-to-network scale differs in x and y ({sw:.3f} vs {sh:.3f}): "
                 "the network input was centre-cropped or padded (upstream --resize_mode crop on "
                 "a portrait frame), so depth does not cover the full frame; rerun with "
                 "--resize_mode fit or pass --allow_crop")
    h, w = (int(x) for x in fz["net_hw"])
    K_orig = fz["K_orig"][row_of]
    K = np.zeros((n, 3, 3))
    K[:, 0, 0] = K_orig[:, 0, 0] / sw
    K[:, 1, 1] = K_orig[:, 1, 1] / sh
    K[:, 0, 2] = K_orig[:, 0, 2] / sw
    K[:, 1, 2] = K_orig[:, 1, 2] / sh
    K[:, 2, 2] = 1.0
    k1 = fz["k1"][row_of].astype(np.float64)

    s_frame = fz["scale"][row_of].astype(np.float64)
    s_pts = fz["scale_points"][row_of]
    s_init = float(fz["init_norm_scale"])
    scale = s_init * s_frame if np.isfinite(s_init) else s_frame
    has_depth = (s_pts != 0) & np.isfinite(scale) & (scale > 0)

    depth_all, conf_all = fz["depth"], fz["conf"]  # decompress once
    depth = np.zeros((n, h, w), np.float32)
    conf = np.zeros((n, h, w), np.float32)
    for i in range(n):
        if not has_depth[i]:
            continue
        r = row_of[i]
        d = np.nan_to_num(depth_all[r].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        d[d < 0] = 0.0
        c = np.clip(np.nan_to_num(conf_all[r].astype(np.float32), nan=0.0), 0.0, 1.0)
        depth[i], conf[i] = undistort_resample(d, c, K[i], float(k1[i]))

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out_dir, "glob3r_raw.npz"),
        files=np.array(files), rec=np.array(rec, np.uint8),
        cam2world=c2w.astype(np.float32), K=K.astype(np.float32),
        depth=depth, conf=conf, has_depth=has_depth, scale=scale.astype(np.float32),
        is_keyframe=fz["is_keyframe"][row_of], k1=k1.astype(np.float32), scale_points=s_pts,
    )
    t_convert = time.time() - t0

    info = {}
    if args.run_info:
        with open(args.run_info) as f:
            info = json.load(f)
    infer_args = info.get("infer_args", [])
    peak = info.get("peak_vram_gb")
    torch_peak = (float(fz["torch_peak_reserved_gb"]) if "torch_peak_reserved_gb" in fz.files
                  else None)
    good = scale[has_depth]
    meta = {
        "mode": "unordered" if "--unordered" in infer_args else "ordered",
        "infer_args": infer_args,
        "n_frames": n,
        "n_with_depth": int(has_depth.sum()),
        "resolution": [w, h],
        "runtime_s": {"inference": info.get("inference_s"), "convert": round(t_convert, 2)},
        "peak_vram_gb": peak if peak is not None else torch_peak,
        "glob3r_commit": info.get("glob3r_commit"),
        "notes": (
            "cam2world = inverse of Glob3R's final-BA world-to-camera (OpenCV axes), rows mapped "
            "from Glob3R's processing order to index.json order by file name and checked against "
            f"poses_final.txt (max abs diff {pose_err:.1e}). Units are Glob3R's normalised frame "
            "(camera-centre 10-90 percentile bbox diagonal = 10), NOT metres: scale[i] = "
            "depth[i] / raw Pi3X metric z-depth, so metres = Glob3R units / median(scale) "
            f"(median {float(np.median(good)) if good.size else float('nan'):.4g}, "
            f"spread p10-p90 {np.percentile(good, 10) if good.size else float('nan'):.4g}-"
            f"{np.percentile(good, 90) if good.size else float('nan'):.4g}). "
            "depth/conf: every frame gets Pi3X dense depth from one sliding window (not only "
            "Glob3R keyframes; is_keyframe marks those), rescaled per frame by a RANSAC fit to the "
            "BA points; has_depth=False where that fit had no BA points (scale_points=0). Camera "
            f"is SIMPLE_RADIAL (shared focal, k1={float(np.median(k1)):.4g}); depth/conf were "
            "resampled onto the undistorted pinhole grid of K. conf = sigmoid of Pi3X's confidence "
            "logit (upstream fuses conf>0.25 and non-edge pixels into points_dense.ply). K uses "
            "integer pixel indices with principal point (W/2, H/2)."
        ),
        "peak_vram_gb_torch_reserved": torch_peak,
        "peak_vram_gb_device_delta": peak,
        "peak_rss_gb": info.get("peak_rss_gb"),
        "n_keyframes": int(fz["is_keyframe"].sum()),
        "init_norm_scale": s_init,
        "median_scale": float(np.median(good)) if good.size else None,
        "focal_px": float(np.median(K[:, 0, 0])),
        "k1": float(np.median(k1)),
        "pose_check_max_abs_diff": pose_err,
    }
    with open(os.path.join(args.out_dir, "glob3r_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out_dir}/glob3r_raw.npz ({n} frames, {int(has_depth.sum())} with depth, "
          f"{w}x{h}) in {t_convert:.1f}s")


if __name__ == "__main__":
    main()
