"""M1 ingest: decode phone videos and select sharp, non-redundant keyframes.

Rotation (PyAV 19.0.1 with its bundled FFmpeg 9.0.2 libraries): phones store frames in
sensor orientation plus a display matrix in the container (MP4/MOV `tkhd`). PyAV 19 has
no `Stream.side_data` and the stream metadata has no `rotate` tag; the matrix is only
exposed as raw bytes in `stream.codec_context.coded_side_data["display_matrix"]`. The
decoder, however, attaches it to every frame (`AV_FRAME_DATA_DISPLAYMATRIX`), and
`VideoFrame.rotation` returns its angle (`av_display_rotation_get`, counter-clockwise
degrees as an int; -90 for a portrait iPhone video). We use that per-frame angle and
rotate the array clockwise by `-rotation`, rounded to a quarter turn, with `np.rot90`:
the same rule as the ffmpeg CLI's autorotate and OpenCV's `CAP_PROP_ORIENTATION_AUTO`.
Checked on a real portrait HEVC phone video with a -90 degree matrix (SceneDiff
`living_room_1_living_room_2`): the frames match the ffmpeg CLI, OpenCV and the
benchmark's own upright re-encode. The tests cover 0/90/180/270 degrees written with
`ffmpeg -display_rotation`. Mirroring in the matrix is ignored (phones do not write it).

Two keyframe modes (`ingest.mode`): `sharp` (default, `select_keyframes`) and `fixed`
(`select_fixed_keyframes`): the decoded frame closest to every multiple of
`ingest.fixed_interval_s`, as needed for the SceneDiff benchmark (ground truth at 1 fps).

If one video is portrait and the other landscape, `reconcile_orientation` centre-crops the
keyframes of both to squares (`square_crop` in index.json: the side, or 0): one Pi3X pass
needs one image size. Each recording loses the two margins of its long side (44 % of its
pixels for 16:9) but stays upright; turning one recording a quarter turn instead keeps
every pixel, but Pi3X then reconstructs the turned recording with a slight systematic
offset (a table top came out as "added"), and its crops are sideways for the VLM.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from PIL import Image

from roomdiff.paths import RECORDINGS, OutputPaths

SHARPNESS_SIDE = 512  # longer side (px) of the image used for the sharpness measure
MOTION_SIDE = 160  # longer side (px) of the thumbnails used for the motion measure
MIN_KEEP_FRACTION = 0.6  # never return fewer than this fraction of num_keyframes


@dataclass
class VideoInfo:
    """Basic properties of a video; width and height are after the rotation fix."""

    path: str
    width: int
    height: int
    fps: float
    num_frames: int
    duration: float  # seconds
    rotation: int  # clockwise degrees applied to make frames upright (iPhone portrait: 90)
    backend: str  # "pyav" or "opencv"


@dataclass
class Keyframe:
    """One selected video frame."""

    index: int  # frame index in the source video
    timestamp: float  # seconds from the start of the video
    image: np.ndarray = field(repr=False)  # upright RGB uint8, longer side <= max_side
    sharpness: float
    t_target: float | None = None  # fixed mode: intended time k * interval (s)
    benchmark_frame: int | None = None  # fixed mode: SceneDiff frame k (t_target = k s)


# --------------------------------------------------------------------------- decoding


def probe_video(path: str | Path) -> VideoInfo:
    """Read resolution, frame rate, frame count, duration and rotation of a video."""
    try:
        container = av.open(str(path))
    except av.FFmpegError:
        return _probe_opencv(path)
    with container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 30.0)
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = (container.duration or 0) / av.time_base
        num_frames = stream.frames or round(duration * fps)  # MP4/MOV store the exact count
        first = next(container.decode(stream), None)
        if first is None:
            raise OSError(f"no decodable video frames in {path}")
        rotation = _rotation_cw(first)
        width, height = first.width, first.height
    if rotation in (90, 270):
        width, height = height, width
    return VideoInfo(
        str(Path(path).resolve()), width, height, fps, num_frames, duration, rotation,
        "pyav",
    )


def iter_video_frames(
    path: str | Path, every_n: int = 1
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield `(frame_index, timestamp_s, upright RGB uint8 HxWx3)` for every n-th frame.

    Every frame is decoded (inter-frame codecs require it) but only every n-th one is
    converted to RGB. Uses PyAV; falls back to OpenCV if PyAV cannot open the file.
    """
    try:
        container = av.open(str(path))
    except av.FFmpegError:
        yield from _iter_frames_opencv(path, every_n)
        return
    with container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # multi-threaded decoding
        fps = float(stream.average_rate or 30.0)
        start = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
        for index, frame in enumerate(container.decode(stream)):
            if index % every_n:
                continue
            t = float(frame.time) - start if frame.time is not None else index / fps
            rgb = frame.to_ndarray(format="rgb24")
            yield index, t, _upright(rgb, _rotation_cw(frame))


def _rotation_cw(frame: av.VideoFrame) -> int:
    """Clockwise rotation (0/90/180/270) that makes a decoded PyAV frame upright."""
    return round(-frame.rotation / 90) * 90 % 360


def _upright(rgb: np.ndarray, rotation_cw: int) -> np.ndarray:
    if rotation_cw == 0:
        return rgb
    return np.ascontiguousarray(np.rot90(rgb, k=-rotation_cw // 90))  # k < 0: clockwise


def _probe_opencv(path: str | Path) -> VideoInfo:
    cap = _open_opencv(path)
    try:
        ok, bgr = cap.read()  # OpenCV applies the rotation metadata itself
        if not ok:
            raise OSError(f"cannot decode any frame of {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
    finally:
        cap.release()
    height, width = bgr.shape[:2]
    return VideoInfo(
        str(Path(path).resolve()), width, height, fps, num_frames, num_frames / fps,
        rotation, "opencv",
    )


def _iter_frames_opencv(
    path: str | Path, every_n: int
) -> Iterator[tuple[int, float, np.ndarray]]:
    cap = _open_opencv(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        index = 0
        while cap.grab():  # grab() decodes; retrieve() converts only the frames we keep
            if index % every_n == 0:
                ok, bgr = cap.retrieve()
                if ok:
                    yield index, index / fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        cap.release()


def _open_opencv(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path} with PyAV or OpenCV")
    return cap


# --------------------------------------------------------------------- frame scores


def sharpness(frame_rgb: np.ndarray) -> float:
    """Variance of the Laplacian of the grayscale image, downscaled to SHARPNESS_SIDE px
    so that scores are comparable across video resolutions. Higher is sharper."""
    gray = _downscale(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY), SHARPNESS_SIDE)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def motion_thumbnail(frame_rgb: np.ndarray) -> np.ndarray:
    """Small grayscale image (longer side MOTION_SIDE px) used by `motion_score`."""
    return _downscale(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY), MOTION_SIDE)


def motion_score(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Cheap global-motion estimate between two equally sized grayscale uint8 frames.

    Mean absolute difference (intensities in [0, 1]) of the two images downscaled to
    MOTION_SIDE px, after subtracting each image's mean so that a global exposure shift
    does not count as motion. 0 means identical.

    Why: it costs ~0.02 ms per pair and grows with camera motion until the views stop
    overlapping, then plateaus far above the threshold, so large motion is never taken
    for a still camera. Mean scores at gaps of 1/4/15/90 frames: parkour_long
    0.07/0.14/0.19/0.22, SceneDiff living_room_1 0.03/0.06/0.10/0.14. Consecutive frames
    of 33 handheld SceneDiff clips score 0.001 (near rest) to 0.06 (median 0.024), so
    min_motion=0.02 between keyframes, which are several frames apart, only removes
    near-static stretches. The geometric alternative, median Farneback optical flow,
    costs ~0.4 ms and saturates at ~2 % of the image diagonal after 2-4 frames (the
    displacement exceeds its capture range), so it is not monotone over keyframe gaps.
    Caveat: the difference underestimates motion over textureless views (blank wall).
    """
    a = _downscale(prev_gray, MOTION_SIDE).astype(np.float32) / 255.0
    b = _downscale(gray, MOTION_SIDE).astype(np.float32) / 255.0
    return float(np.mean(np.abs((a - a.mean()) - (b - b.mean()))))


def _downscale(image: np.ndarray, long_side: int) -> np.ndarray:
    """Resize so the longer side is at most `long_side` (never upscales)."""
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return image
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ keyframe choice


def candidate_stride(num_frames: int, num_keyframes: int, sharpness_window: int) -> int:
    """Frame stride giving about `num_keyframes * sharpness_window` candidates."""
    return max(1, num_frames // (num_keyframes * sharpness_window))


def select_keyframes(
    video_path: str | Path,
    num_keyframes: int = 48,
    max_side: int = 1024,
    sharpness_window: int = 5,
    min_motion: float = 0.02,
) -> list[Keyframe]:
    """Pick up to `num_keyframes` sharp, non-redundant keyframes spread over the video.

    One decoding pass converts every stride-th frame (about num_keyframes *
    sharpness_window candidates). The timeline is split into `num_keyframes` equal
    segments and the sharpest candidate of each segment is picked. A pick whose
    `motion_score` to the previously kept keyframe is below `min_motion` is skipped
    (camera at rest), but at least 60 % of `num_keyframes` are returned: the skipped
    picks with the most motion are restored first. Motion filters the sharpest pick
    instead of choosing within a segment because blur also raises the photometric
    difference and would favour blurred frames. Images are resized so that the longer
    side is `max_side`; videos smaller than that are not upscaled.
    """
    if num_keyframes < 1 or sharpness_window < 1:
        raise ValueError("num_keyframes and sharpness_window must be positive")
    info = probe_video(video_path)
    total = max(info.num_frames, 1)
    stride = candidate_stride(total, num_keyframes, sharpness_window)

    # Single decoding pass; only the current best of each segment is kept in memory.
    best: dict[int, tuple[Keyframe, np.ndarray]] = {}  # segment -> (keyframe, thumbnail)
    for index, t, rgb in iter_video_frames(video_path, every_n=stride):
        segment = min(index * num_keyframes // total, num_keyframes - 1)
        score = sharpness(rgb)
        if segment not in best or score > best[segment][0].sharpness:
            keyframe = Keyframe(index, t, _downscale(rgb, max_side), score)
            best[segment] = (keyframe, motion_thumbnail(rgb))

    picks = [best[s] for s in sorted(best)]
    min_keep = math.ceil(MIN_KEEP_FRACTION * num_keyframes)
    return _drop_static(picks, min_motion, min_keep)


def fixed_targets(duration: float, interval_s: float) -> list[float]:
    """Target times `k * interval_s` for k = 0, 1, 2, ... while `k * interval_s <= duration`."""
    if interval_s <= 0:
        raise ValueError("fixed_interval_s must be positive")
    count = math.floor(max(duration, 0.0) / interval_s + 1e-6) + 1
    return [k * interval_s for k in range(count)]


def select_fixed_keyframes(
    video_path: str | Path, interval_s: float = 1.0, max_side: int = 1024
) -> list[Keyframe]:
    """One keyframe per target time `t_k = k * interval_s` (k = 0, 1, ... while `t_k <=`
    duration): the decoded frame whose timestamp is closest to `t_k` (ties: the earlier
    one), with no sharpness or motion filtering. Images are resized to `max_side`.

    `benchmark_frame` is `round(t_k)` when `t_k` is a whole second (every keyframe for
    `interval_s = 1`, then it equals k) and None otherwise: the SceneDiff evaluator keeps
    every 30th frame of its 30 fps re-encode, so its frame k shows time k s.
    """
    targets = fixed_targets(probe_video(video_path).duration, interval_s)
    picks: list[Keyframe] = []
    prev: tuple[int, float, np.ndarray] | None = None
    for frame in iter_video_frames(video_path):
        while len(picks) < len(targets) and targets[len(picks)] <= frame[1]:
            target = targets[len(picks)]
            use_prev = prev is not None and target - prev[1] <= frame[1] - target
            picks.append(_fixed_keyframe(prev if use_prev else frame, target, max_side))
        prev = frame
    while prev is not None and len(picks) < len(targets):  # targets after the last frame
        picks.append(_fixed_keyframe(prev, targets[len(picks)], max_side))
    return picks


def _fixed_keyframe(
    frame: tuple[int, float, np.ndarray], target: float, max_side: int
) -> Keyframe:
    index, t, rgb = frame
    whole = abs(target - round(target)) < 1e-6
    return Keyframe(index, t, _downscale(rgb, max_side), sharpness(rgb), target,
                    round(target) if whole else None)


def _drop_static(
    picks: list[tuple[Keyframe, np.ndarray]], min_motion: float, min_keep: int
) -> list[Keyframe]:
    """Drop picks that barely moved since the last kept one; keep at least `min_keep`."""
    kept: list[tuple[Keyframe, np.ndarray]] = []
    dropped: list[tuple[float, Keyframe]] = []
    for keyframe, thumb in picks:
        if kept:
            motion = motion_score(kept[-1][1], thumb)
            if motion < min_motion:
                dropped.append((motion, keyframe))
                continue
        kept.append((keyframe, thumb))

    result = [keyframe for keyframe, _ in kept]
    dropped.sort(key=lambda d: d[0], reverse=True)  # most motion = least redundant
    result += [keyframe for _, keyframe in dropped[: max(0, min_keep - len(result))]]
    return sorted(result, key=lambda k: k.index)


# ------------------------------------------------------------------------- outputs


def write_keyframes(
    keyframes: list[Keyframe],
    out_dir: str | Path,
    prefix: str,
    video: VideoInfo | None = None,
    jpeg_quality: int = 95,
) -> dict[str, Any]:
    """Write `<prefix>_<00000>.jpg, ...` and `index.json` to `out_dir`; return the index.

    The index lists, per keyframe, the file, source frame index, timestamp, sharpness and
    size, plus the source video's path, resolution, fps, duration and rotation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{prefix}_*.jpg"):  # left over from an earlier run
        stale.unlink()

    frames = []
    for i, kf in enumerate(keyframes):
        name = f"{prefix}_{i:05d}.jpg"
        Image.fromarray(kf.image).save(out_dir / name, quality=jpeg_quality)
        height, width = kf.image.shape[:2]
        entry = {
            "file": name,
            "frame_index": kf.index,
            "timestamp": round(kf.timestamp, 4),
            "sharpness": round(kf.sharpness, 2),
            "width": width,
            "height": height,
        }
        if kf.t_target is not None:  # fixed mode
            entry["t_target"] = round(kf.t_target, 4)
            entry["benchmark_frame"] = kf.benchmark_frame
        frames.append(entry)
    record = {"video": asdict(video) if video else None, "frames": frames}
    (out_dir / "index.json").write_text(json.dumps(record, indent=2))
    return record


def _orientation(height: int, width: int) -> str:
    return "portrait" if height > width else "landscape" if width > height else "square"


def _square_crop(image: np.ndarray) -> np.ndarray:
    """The central square of an image (side = its shorter side)."""
    h, w = image.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return np.ascontiguousarray(image[y0:y0 + side, x0:x0 + side])


def reconcile_orientation(before: list[Keyframe], after: list[Keyframe]) -> int:
    """Give both recordings one image shape; return the side of the square crop applied
    to every keyframe of both, or 0 if nothing was done.

    If one recording is portrait (height > width) and the other landscape, every image
    of both is replaced by its central square (side = the shorter side of that
    recording's images), in place. Square images match either orientation.
    """
    if not before or not after:
        return 0
    shapes = {_orientation(*keyframes[0].image.shape[:2]) for keyframes in (before, after)}
    if shapes != {"portrait", "landscape"}:
        return 0
    for keyframe in before + after:
        keyframe.image = _square_crop(keyframe.image)
    return min(before[0].image.shape[:2])


def ingest_pair(
    before_path: str | Path,
    after_path: str | Path,
    out_dir: str | Path,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Select and write keyframes of both videos; write `<out>/keyframes/index.json`.

    `cfg` is the full config; only its `ingest` section is used (`mode`: "sharp", the
    default, or "fixed"). Between selection and writing, `reconcile_orientation` gives
    both recordings one image shape (`square_crop` per recording). Returns the index.
    """
    params = cfg["ingest"]
    mode = params.get("mode", "sharp")
    if mode not in ("sharp", "fixed"):
        raise ValueError(f"ingest.mode must be 'sharp' or 'fixed', got {mode!r}")
    paths = OutputPaths(out_dir)
    start = time.perf_counter()
    index: dict[str, Any] = {"params": dict(params)}

    selected: dict[str, tuple[VideoInfo, list[Keyframe], float]] = {}
    for name, video_path in zip(RECORDINGS, (before_path, after_path)):
        t0 = time.perf_counter()
        info = probe_video(video_path)
        if mode == "fixed":
            keyframes = select_fixed_keyframes(
                video_path, params["fixed_interval_s"], params["max_side"]
            )
        else:
            keyframes = select_keyframes(
                video_path,
                num_keyframes=params["num_keyframes"],
                max_side=params["max_side"],
                sharpness_window=params["sharpness_window"],
                min_motion=params["min_motion"],
            )
        seconds = time.perf_counter() - t0
        selected[name] = (info, keyframes, seconds)

        if mode == "fixed":
            selection = f"one frame every {params['fixed_interval_s']} s"
        else:
            stride = candidate_stride(
                max(info.num_frames, 1), params["num_keyframes"], params["sharpness_window"]
            )
            selection = f"{math.ceil(info.num_frames / stride)} candidates"
        mean_sharpness = float(np.mean([k.sharpness for k in keyframes]))
        print(
            f"[ingest] {name}: {info.num_frames} frames decoded "
            f"({info.width}x{info.height}, {info.fps:.2f} fps, {info.duration:.1f} s, "
            f"rotation {info.rotation} deg, {info.backend}), {selection}, "
            f"{len(keyframes)} keyframes kept, mean sharpness {mean_sharpness:.1f}, "
            f"{seconds:.1f} s"
        )

    square = reconcile_orientation(selected["before"][1], selected["after"][1])
    if square:
        b, a = selected["before"][0], selected["after"][0]
        print(
            f"[ingest] WARNING: before {_orientation(b.height, b.width)} {b.width}x{b.height}, "
            f"after {_orientation(a.height, a.width)} {a.width}x{a.height}: mixed orientation, "
            f"the keyframes of both are centre-cropped to {square}x{square} squares (one Pi3X "
            "pass needs one image size); each recording loses its long-side margins"
        )
    for name, (info, keyframes, seconds) in selected.items():
        t0 = time.perf_counter()
        record = write_keyframes(
            keyframes, paths.keyframes_of(name), name, info, params["jpeg_quality"]
        )
        seconds += time.perf_counter() - t0
        index[name] = {
            "dir": name,
            "wall_time_s": round(seconds, 2),
            "square_crop": min(keyframes[0].image.shape[:2]) if square and keyframes else 0,
            **record,
        }

    index["wall_time_s"] = round(time.perf_counter() - start, 2)
    paths.keyframes_index.write_text(json.dumps(index, indent=2))
    print(f"[ingest] wrote {paths.keyframes_index} ({index['wall_time_s']:.1f} s total)")
    return index
