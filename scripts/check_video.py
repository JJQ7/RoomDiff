"""Capture-day check of a before / after video pair: run it before leaving the room.

Usage: python scripts/check_video.py before.mov after.mov [--json out.json]

One PASS / WARN / FAIL line per check and video:
* readable: PyAV (or OpenCV) decodes the file;
* duration: 30-60 s wanted, under 15 s fails;
* orientation: upright size and rotation metadata (display matrix); portrait warns;
* frame rate and frame count (24-60 fps wanted);
* codec: HEVC warns (the protocol asks for H.264, HEVC decodes fine); 10-bit, HLG / PQ,
  BT.2020 or Dolby Vision (iPhone "HDR Video") warns: HDR shifts colours between takes;
* sharpness of ~60 evenly spaced frames with `roomdiff.ingest.sharpness` (variance of the
  Laplacian at 512 px). It grows with the scene's texture (a blank wall scores low when
  sharp), so a frame counts as blurry below 0.4 x the video's own median;
* keyframes: the real selection of `roomdiff.ingest` with the defaults of
  configs/default.yaml. Usable = not blurry and not one of the picks that ingest only keeps
  to reach 60 % of `num_keyframes` while the camera was at rest. Fewer than 24 fails;
* motion: `roomdiff.ingest.motion_score` (mean absolute difference of mean-free
  thumbnails) between consecutive samples, about one keyframe interval apart; above 0.2
  the two views barely overlap (fast pan): warns if more than 10 % of the pairs are.
Pair: both videos need the same upright aspect ratio (one Pi3X pass takes one image size).

Exit status 1 if any check fails (unreadable, < 15 s, mixed orientation, < 24 usable
keyframes), else 0. `--json` also writes every number to a file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # roomdiff without installing
from roomdiff.config import load_config
from roomdiff.ingest import (
    Keyframe,
    iter_video_frames,
    motion_score,
    motion_thumbnail,
    probe_video,
    select_fixed_keyframes,
    select_keyframes,
    sharpness,
)

N_SAMPLES = 60  # evenly spaced frames for the sharpness and motion statistics
MIN_DURATION_S = 15.0
GOOD_DURATION_S = (30.0, 60.0)
GOOD_FPS = (23.5, 60.5)
BLUR_REL = 0.4  # blurry: sharpness below this fraction of the video's median
MAX_BLURRY = 0.2  # warn above this fraction of blurry samples
SOFT_MEDIAN = 50.0  # median sharpness below this: soft overall (focus, dirty lens, dark room)
FAST_MOTION = 0.2  # motion_score of consecutive samples above this: views barely overlap
MAX_FAST = 0.1  # warn above this fraction of fast pairs
MIN_USABLE, OK_USABLE = 24, 32  # usable keyframes: fail below the first, warn below the second
ASPECT_TOL = 0.01  # the rule of roomdiff.recon.preprocess_images
THUMB_SIDE = 320  # keyframe images are only needed for motion thumbnails (160 px)
HDR_TRANSFER = {16: "PQ", 18: "HLG"}  # FFmpeg AVColorTransferCharacteristic
BT2020 = 9  # FFmpeg AVColorPrimaries
DECODE_ERRORS = (OSError, ValueError, RuntimeError, av.FFmpegError)
RECORDINGS = ("before", "after")


def codec_info(path: str) -> dict[str, Any]:
    """Codec, profile, pixel format, bit depth and HDR signalling of the video stream."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        cc = stream.codec_context
        fmt = cc.format
        return {
            "codec": cc.name, "profile": cc.profile, "pix_fmt": fmt.name if fmt else None,
            "bits": max((c.bits for c in fmt.components), default=8) if fmt else 8,
            "transfer": HDR_TRANSFER.get(cc.color_trc), "bt2020": cc.color_primaries == BT2020,
            "dolby_vision": "dovi_conf" in cc.coded_side_data,
            "bit_rate": stream.bit_rate or container.bit_rate,
        }


def sample_frames(path: str, num_frames: int) -> dict[str, Any]:
    """Sharpness of ~N_SAMPLES evenly spaced frames and the motion between neighbours."""
    step = max(1, -(-num_frames // N_SAMPLES))  # ceil: at most N_SAMPLES frames
    times: list[float] = []
    sharp: list[float] = []
    motion: list[float] = []
    previous = None
    for _, t, rgb in iter_video_frames(path, every_n=step):
        thumb = motion_thumbnail(rgb)
        if previous is not None:
            motion.append(motion_score(previous, thumb))
        times.append(t)
        sharp.append(sharpness(rgb))
        previous = thumb
    return {"step": step, "times": times, "sharpness": sharp, "motion": motion}


def at_rest(keyframes: list[Keyframe], min_motion: float) -> list[bool]:
    """Picks that barely moved since the last moving pick: the greedy rule of the motion
    filter of `select_keyframes`, so these are the picks it only kept to reach its minimum."""
    flags, last = [], None
    for keyframe in keyframes:
        thumb = motion_thumbnail(keyframe.image)
        still = last is not None and motion_score(last, thumb) < min_motion
        flags.append(still)
        if not still:
            last = thumb
    return flags


def keyframe_stats(path: str, params: dict[str, Any], blur_below: float) -> dict[str, Any]:
    """Keyframes the default ingest settings select, and how many of them are usable."""
    mode = params.get("mode", "sharp")
    if mode == "fixed":
        keyframes = select_fixed_keyframes(path, params["fixed_interval_s"], THUMB_SIDE)
        still = [False] * len(keyframes)
    else:
        keyframes = select_keyframes(
            path, num_keyframes=params["num_keyframes"], max_side=THUMB_SIDE,
            sharpness_window=params["sharpness_window"], min_motion=params["min_motion"])
        still = at_rest(keyframes, params["min_motion"])
    blurry = [k.sharpness < blur_below for k in keyframes]
    return {"mode": mode, "wanted": params["num_keyframes"] if mode != "fixed" else None,
            "selected": len(keyframes), "at_rest": int(sum(still)),
            "blurry": int(sum(b and not s for b, s in zip(blurry, still))),
            "usable": int(sum(not (b or s) for b, s in zip(blurry, still)))}


def grade(fail: bool, warn: bool) -> str:
    return "FAIL" if fail else "WARN" if warn else "PASS"


def check_video(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """All per-video checks; `checks` is a list of {check, status, message}."""
    result: dict[str, Any] = {"path": path, "checks": []}

    def add(name: str, status: str, message: str) -> None:
        result["checks"].append({"check": name, "status": status, "message": message})

    try:
        info = probe_video(path)
        codec = codec_info(path) if info.backend == "pyav" else None
        samples = sample_frames(path, info.num_frames)
        if not samples["times"]:
            raise OSError("no frame decoded")
    except DECODE_ERRORS as exc:
        add("readable", "FAIL", f"cannot decode: {exc}")
        return result
    result.update(info=asdict(info), codec=codec, samples=samples)
    add("readable", "PASS", f"decoded with {info.backend}")

    lo, hi = GOOD_DURATION_S
    good, short = lo <= info.duration <= hi, info.duration < MIN_DURATION_S
    message = f"{info.duration:.1f} s"
    if short:
        message += f": shorter than {MIN_DURATION_S:.0f} s"
    if not good:
        message += f" (want {lo:.0f}-{hi:.0f} s)"
    add("duration", grade(short, not good), message)

    portrait = info.height > info.width
    stored = (f"stored {info.height}x{info.width}, " if info.rotation in (90, 270) else "")
    display = -info.rotation + (360 if info.rotation > 180 else 0)  # ffprobe's angle, ccw
    message = (f"{'portrait' if portrait else 'landscape'} {info.width}x{info.height} "
               f"({stored}display rotation {display} deg)")
    add("orientation", "WARN" if portrait else "PASS",
        message + (": hold the phone landscape" if portrait else ""))

    message = f"{info.fps:.2f} fps, {info.num_frames} frames"
    add("frame rate", "PASS" if GOOD_FPS[0] <= info.fps <= GOOD_FPS[1] else "WARN", message)

    if codec is None:
        add("codec", "WARN", "unknown (PyAV cannot open the file; OpenCV fallback)")
    else:
        message = f"{codec['codec']} {codec['profile']}, {codec['bits']}-bit {codec['pix_fmt']}"
        hdr = [x for x, on in ((f"{codec['bits']}-bit", codec["bits"] > 8),
                               (codec["transfer"], codec["transfer"] is not None),
                               ("BT.2020", codec["bt2020"]),
                               ("Dolby Vision", codec["dolby_vision"])) if on]
        if hdr:
            add("codec", "WARN", f"{message}: HDR ({', '.join(hdr)}) shifts colours between "
                "takes; turn off Settings > Camera > Record Video > HDR Video")
        elif codec["codec"] == "hevc":
            add("codec", "WARN", f"{message}: protocol asks for H.264 (Settings > Camera > "
                "Formats > Most Compatible); HEVC decodes fine")
        else:
            add("codec", "PASS", message)

    sharp = np.asarray(samples["sharpness"])
    median = float(np.median(sharp))
    blur_below = BLUR_REL * median
    blurry = float(np.mean(sharp < blur_below))
    p10, p90 = np.percentile(sharp, [10, 90])
    message = (f"{len(sharp)} samples: median {median:.0f}, p10 {p10:.0f}, p90 {p90:.0f}; "
               f"{blurry:.0%} blurry (< {blur_below:.0f} = {BLUR_REL} x median)")
    if median < SOFT_MEDIAN:
        message += "; soft overall: check focus, lens and light"
    add("sharpness", grade(False, blurry > MAX_BLURRY or median < SOFT_MEDIAN), message)
    result["blurry_fraction"] = blurry

    try:
        stats = keyframe_stats(path, params, blur_below)
    except DECODE_ERRORS as exc:
        add("keyframes", "FAIL", f"keyframe selection failed: {exc}")
    else:
        result["keyframes"] = stats
        wanted = f", {stats['wanted']} wanted" if stats["wanted"] else ""
        message = (f"{stats['selected']} picked by ingest ({stats['mode']} mode{wanted}), "
                   f"{stats['usable']} usable ({stats['at_rest']} at rest, "
                   f"{stats['blurry']} blurry)")
        add("keyframes", grade(stats["usable"] < MIN_USABLE, stats["usable"] < OK_USABLE),
            message)

    motion = np.asarray(samples["motion"])
    if len(motion):
        fast = motion > FAST_MOTION
        gap = samples["step"] / info.fps
        message = (f"samples {gap:.2f} s apart: median {np.median(motion):.3f}, p90 "
                   f"{np.percentile(motion, 90):.3f}, max {motion.max():.3f}; "
                   f"{fast.mean():.0%} above {FAST_MOTION} (fast pan)")
        if fast.mean() > MAX_FAST:
            worst = np.argsort(-motion)[: min(5, int(fast.sum()))]
            when = ", ".join(f"{samples['times'][i + 1]:.1f} s" for i in sorted(worst))
            message += f" at {when}: turn more slowly"
        add("motion", "WARN" if fast.mean() > MAX_FAST else "PASS", message)
        result["fast_fraction"] = float(fast.mean())
    return result


def check_pair(before: dict[str, Any], after: dict[str, Any]) -> dict[str, str]:
    """Both recordings must have the same upright aspect ratio (one Pi3X image size)."""
    def describe(info: dict[str, Any]) -> str:
        shape = "portrait" if info["height"] > info["width"] else "landscape"
        return f"{shape} {info['width']}x{info['height']}"

    ra, rb = (i["width"] / i["height"] for i in (before, after))
    if abs(ra - rb) > ASPECT_TOL * max(ra, rb):
        return {"check": "orientation", "status": "FAIL", "message":
                f"before {describe(before)}, after {describe(after)}: mixed orientation; "
                "record both videos the same way (one Pi3X pass needs one image size)"}
    if (before["width"], before["height"]) != (after["width"], after["height"]):
        return {"check": "orientation", "status": "WARN", "message":
                f"before {describe(before)}, after {describe(after)}: same orientation, "
                "different resolutions (camera mode changed?)"}
    return {"check": "orientation", "status": "PASS", "message": f"both {describe(before)}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", help="video of the room before the changes")
    parser.add_argument("after", help="video of the room after the changes")
    parser.add_argument("--json", type=Path, metavar="OUT", help="also write the results here")
    args = parser.parse_args(argv)
    params = dict(load_config()["ingest"])

    report: dict[str, Any] = {"videos": {}, "pair": []}
    for name, path in zip(RECORDINGS, (args.before, args.after)):
        print(f"{name}: {path}", flush=True)
        result = check_video(str(path), params)
        report["videos"][name] = result
        for c in result["checks"]:
            print(f"  {c['status']:<4}  {c['check']:<11}  {c['message']}")
    infos = [report["videos"][name].get("info") for name in RECORDINGS]
    if all(infos):
        report["pair"].append(check_pair(*infos))
        print("pair:")
        for c in report["pair"]:
            print(f"  {c['status']:<4}  {c['check']:<11}  {c['message']}")

    statuses = [c["status"] for v in report["videos"].values() for c in v["checks"]]
    statuses += [c["status"] for c in report["pair"]]
    fails, warns = statuses.count("FAIL"), statuses.count("WARN")
    report["result"] = "FAIL" if fails else "WARN" if warns else "PASS"
    print(f"RESULT: {report['result']} ({fails} failed, {warns} warnings)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
