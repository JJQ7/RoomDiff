"""Tests for M1 ingest on tiny synthetic videos written with imageio-ffmpeg's ffmpeg."""

import itertools
import json
import math
import subprocess
from pathlib import Path

import av
import cv2
import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from roomdiff import ingest
from roomdiff.cli import app
from roomdiff.config import load_config
from roomdiff.ingest import (
    candidate_stride,
    fixed_targets,
    ingest_pair,
    iter_video_frames,
    motion_score,
    motion_thumbnail,
    probe_video,
    select_fixed_keyframes,
    select_keyframes,
    sharpness,
)

# ------------------------------------------------------------------------- helpers


def write_video(path: Path, frames: list[np.ndarray], rotation_ccw: int = 0) -> Path:
    """Encode RGB frames as near-lossless H.264 at 30 fps.

    A non-zero `rotation_ccw` is then stored as a display matrix by remuxing with
    `ffmpeg -display_rotation`, which is how phones tag portrait video (iPhone: -90).
    """
    height, width = frames[0].shape[:2]
    plain = path.with_suffix(".plain.mp4") if rotation_ccw else path
    writer = imageio_ffmpeg.write_frames(
        str(plain), (width, height), fps=30, codec="libx264", quality=None,
        output_params=["-crf", "8"], ffmpeg_log_level="error",
    )
    writer.send(None)  # start ffmpeg
    for frame in frames:
        writer.send(np.ascontiguousarray(frame))
    writer.close()
    if rotation_ccw:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-display_rotation", str(rotation_ccw),
             "-i", str(plain), "-c", "copy", str(path)],
            check=True,
        )
    return path


def texture(height: int, width: int, seed: int = 0, smooth: bool = False) -> np.ndarray:
    """Random RGB texture: 4 px blocks with hard edges, or smooth 16 px blobs."""
    block, interpolation = (16, cv2.INTER_LINEAR) if smooth else (4, cv2.INTER_NEAREST)
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (height // block, width // block, 3), dtype=np.uint8)
    return cv2.resize(small, (width, height), interpolation=interpolation)


def marker_frame() -> np.ndarray:
    """Landscape frame with a red marker in the top-left corner."""
    frame = np.full((64, 96, 3), 40, np.uint8)
    frame[4:20, 4:28] = (255, 0, 0)
    return frame


def red_centroid(rgb: np.ndarray) -> tuple[float, float]:
    """Normalised (y, x) centre of the red marker."""
    ys, xs = np.nonzero((rgb[..., 0] > 180) & (rgb[..., 1] < 100))
    return ys.mean() / rgb.shape[0], xs.mean() / rgb.shape[1]


@pytest.fixture(scope="module")
def pan_then_still_video(tmp_path_factory) -> Path:
    """240 frames, 96x128: the camera pans 4 px/frame for 120 frames, then holds still.

    Every frame is blurred except those with index % 20 == 8 (one per 20-frame segment).
    """
    rng = np.random.default_rng(1)
    scene = texture(96, 1200)
    frames = []
    for i in range(240):
        x = 4 * min(i, 119)
        frame = scene[:, x : x + 128].copy()
        noise = rng.integers(-3, 4, frame.shape)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if i % 20 != 8:
            frame = cv2.GaussianBlur(frame, (0, 0), 2.5)
        frames.append(frame)
    return write_video(tmp_path_factory.mktemp("video") / "pan.mp4", frames)


# ------------------------------------------------------------------------ decoding


@pytest.mark.parametrize(
    "rotation_ccw, expected_cw",
    [(0, 0), (-90, 90), (90, 270), (180, 180)],  # -90 CCW is how iPhones tag portrait
)
def test_rotation_metadata_is_applied(tmp_path, rotation_ccw, expected_cw):
    frame = marker_frame()
    path = write_video(tmp_path / "rot.mp4", [frame] * 3, rotation_ccw)

    info = probe_video(path)
    assert info.rotation == expected_cw
    _, _, rgb = next(iter_video_frames(path))
    expected = np.rot90(frame, k=-expected_cw // 90)
    assert rgb.shape == expected.shape == (info.height, info.width, 3)
    assert red_centroid(rgb) == pytest.approx(red_centroid(expected), abs=0.05)

    ok, bgr = cv2.VideoCapture(str(path)).read()  # OpenCV auto-rotates: must agree
    assert ok and red_centroid(bgr[..., ::-1]) == pytest.approx(red_centroid(rgb), abs=0.05)


def test_opencv_fallback_when_pyav_cannot_open(tmp_path, monkeypatch):
    path = write_video(tmp_path / "portrait.mp4", [marker_frame()] * 6, rotation_ccw=-90)
    reference = list(iter_video_frames(path, every_n=2))

    def refuse(*args, **kwargs):
        raise av.FFmpegError(1, "forced failure")

    monkeypatch.setattr(ingest.av, "open", refuse)
    info = probe_video(path)
    frames = list(iter_video_frames(path, every_n=2))
    assert info.backend == "opencv" and info.rotation == 90
    assert [i for i, _, _ in frames] == [i for i, _, _ in reference] == [0, 2, 4]
    assert frames[0][2].shape == reference[0][2].shape == (96, 64, 3)
    centroid = red_centroid(reference[0][2])
    assert red_centroid(frames[0][2]) == pytest.approx(centroid, abs=0.05)


def test_iter_video_frames_stride_and_timestamps(pan_then_still_video):
    frames = list(iter_video_frames(pan_then_still_video, every_n=10))
    assert [i for i, _, _ in frames] == list(range(0, 240, 10))
    assert frames[3][1] == pytest.approx(30 / 30.0, abs=1e-3)
    assert frames[0][2].dtype == np.uint8 and frames[0][2].shape == (96, 128, 3)


# -------------------------------------------------------------------------- scores


def test_sharpness_prefers_sharp_images():
    image = texture(96, 128)
    assert sharpness(image) > 3 * sharpness(cv2.GaussianBlur(image, (0, 0), 2.0))


def test_motion_score_grows_with_shift_and_ignores_exposure():
    scene = texture(96, 400, smooth=True)
    base = motion_thumbnail(scene[:, :128])
    shifts = (0, 1, 2, 4, 8, 16)
    scores = [motion_score(base, motion_thumbnail(scene[:, s : s + 128])) for s in shifts]
    assert scores[0] == pytest.approx(0.0)
    assert scores == sorted(scores) and scores[4] > 0.02
    brighter = np.clip(scene[:, :128].astype(np.int16) + 20, 0, 255).astype(np.uint8)
    assert motion_score(base, motion_thumbnail(brighter)) < 0.01


# ----------------------------------------------------------------------- selection


def test_candidate_stride():
    assert candidate_stride(1800, 48, 5) == 7
    assert candidate_stride(100, 48, 5) == 1


def test_select_picks_sharpest_frame_of_each_segment(pan_then_still_video):
    kfs = select_keyframes(pan_then_still_video, 12, 1024, 5, min_motion=0.0)
    assert [k.index for k in kfs] == list(range(8, 240, 20))
    assert all(k.image.shape == (96, 128, 3) for k in kfs)  # never upscaled
    assert kfs[1].timestamp == pytest.approx(28 / 30.0, abs=1e-3)


def test_select_drops_static_frames_but_keeps_60_percent(pan_then_still_video):
    kfs = select_keyframes(pan_then_still_video, 12, 1024, 5, min_motion=0.02)
    # 6 moving segments + the first still one survive; the other 5 still picks are
    # near-duplicates, and one of them is restored to reach ceil(0.6 * 12) = 8.
    assert len(kfs) == math.ceil(0.6 * 12) == 8
    assert [k.index for k in kfs][:7] == list(range(8, 140, 20))
    assert all(a.index < b.index for a, b in itertools.pairwise(kfs))


def test_select_on_a_static_video_returns_60_percent(tmp_path):
    path = write_video(tmp_path / "still.mp4", [texture(64, 96)] * 60)
    assert len(select_keyframes(path, 10, 1024, 3, min_motion=0.02)) == 6


def test_select_resizes_to_max_side(pan_then_still_video):
    kfs = select_keyframes(pan_then_still_video, 4, 64, 5, min_motion=0.0)
    assert len(kfs) == 4
    assert all(max(k.image.shape[:2]) == 64 for k in kfs)


def grey_ramp_video(path: Path, n: int = 100) -> Path:
    """`n` frames at 30 fps whose uniform grey level encodes the frame index (20 + 2 i)."""
    return write_video(path, [np.full((48, 64, 3), 20 + 2 * i, np.uint8) for i in range(n)])


def test_fixed_targets():
    assert fixed_targets(3.0, 1.0) == [0.0, 1.0, 2.0, 3.0]
    assert fixed_targets(2.99, 1.0) == [0.0, 1.0, 2.0]
    assert fixed_targets(1.2, 0.5) == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError):
        fixed_targets(3.0, 0.0)


def test_select_fixed_keyframes_takes_the_frame_closest_to_each_target(tmp_path):
    path = grey_ramp_video(tmp_path / "ramp.mp4")  # 100 frames = 3.33 s, static content
    kfs = select_fixed_keyframes(path, 1.0, 1024)
    assert [k.index for k in kfs] == [0, 30, 60, 90]  # no sharpness / motion filtering
    assert [k.t_target for k in kfs] == [0.0, 1.0, 2.0, 3.0]
    assert [k.benchmark_frame for k in kfs] == [0, 1, 2, 3]
    for k in kfs:
        assert k.timestamp == pytest.approx(k.t_target, abs=1 / 60)
        assert k.image.shape == (48, 64, 3)
        assert k.image.mean() == pytest.approx(20 + 2 * k.index, abs=2)

    half = select_fixed_keyframes(path, 0.5, 32)
    assert [k.index for k in half] == list(range(0, 91, 15))
    assert [k.benchmark_frame for k in half] == [0, None, 1, None, 2, None, 3]
    assert all(max(k.image.shape[:2]) == 32 for k in half)


def test_ingest_pair_fixed_mode_records_benchmark_frames(tmp_path):
    video = grey_ramp_video(tmp_path / "ramp.mp4", n=70)  # 2.33 s
    cfg = load_config(overrides=["ingest.mode=fixed", "ingest.fixed_interval_s=1.0"])
    index = ingest_pair(video, video, tmp_path / "out", cfg)
    for name in ("before", "after"):
        frames = index[name]["frames"]
        assert [f["frame_index"] for f in frames] == [0, 30, 60]
        assert [f["benchmark_frame"] for f in frames] == [0, 1, 2]
        assert [f["t_target"] for f in frames] == [0.0, 1.0, 2.0]
        assert len(list((tmp_path / "out" / "keyframes" / name).glob("*.jpg"))) == 3
    with pytest.raises(ValueError, match=r"ingest\.mode"):
        ingest_pair(video, video, tmp_path / "bad", load_config(overrides=["ingest.mode=every"]))


# ------------------------------------------------------------------------- outputs


def test_ingest_pair_writes_jpegs_and_index(tmp_path, pan_then_still_video):
    frames = [texture(64, 96, seed) for seed in range(30)]
    after = write_video(tmp_path / "after.mp4", frames, rotation_ccw=-90)
    cfg = load_config(overrides=["ingest.num_keyframes=6", "ingest.min_motion=0.0"])
    index = ingest_pair(pan_then_still_video, after, tmp_path / "out", cfg)

    keyframes_dir = tmp_path / "out" / "keyframes"
    on_disk = json.loads((keyframes_dir / "index.json").read_text())
    assert on_disk == index
    assert index["params"]["num_keyframes"] == 6
    for name in ("before", "after"):
        record = index[name]
        assert len(record["frames"]) == 6
        assert sorted(p.name for p in (keyframes_dir / name).glob("*.jpg")) == [
            f"{name}_{i:05d}.jpg" for i in range(6)
        ]
        first = record["frames"][0]
        fields = {"file", "frame_index", "timestamp", "sharpness", "width", "height"}
        assert set(first) == fields
        image = cv2.imread(str(keyframes_dir / record["dir"] / first["file"]))
        assert image.shape[:2] == (first["height"], first["width"])
        assert json.loads((keyframes_dir / name / "index.json").read_text()) == {
            "video": record["video"], "frames": record["frames"]
        }
    video = index["after"]["video"]
    assert (video["rotation"], video["width"], video["height"]) == (90, 64, 96)
    assert video["fps"] == pytest.approx(30.0) and video["duration"] == pytest.approx(1.0)


def test_cli_ingest_and_eval_arguments(tmp_path, pan_then_still_video):
    runner, video, out = CliRunner(), str(pan_then_still_video), tmp_path / "out"
    args = ["--before", video, "--after", video, "--out", str(out)]
    overrides = ["--set", "ingest.num_keyframes=4", "-s", "ingest.min_motion=0"]
    result = runner.invoke(app, ["ingest", *args, *overrides])
    assert result.exit_code == 0, result.output
    index = json.loads((out / "keyframes" / "index.json").read_text())
    assert index["params"]["min_motion"] == 0.0 and len(index["after"]["frames"]) == 4

    result = runner.invoke(app, ["ingest", *args, "--set", "ingest.num_keyframe=4"])
    assert result.exit_code == 2 and "num_keyframe" in result.output  # typo is reported

    result = runner.invoke(app, ["eval"])  # needs --pairs or --split
    assert result.exit_code == 2 and "--pairs" in result.output


def test_reconcile_orientation_crops_both_recordings_to_squares():
    def keyframe(height: int, width: int) -> ingest.Keyframe:
        image = np.zeros((height, width, 3), np.uint8)
        image[height // 2, width // 2] = 255  # centre marker
        return ingest.Keyframe(0, 0.0, image, 1.0)

    before, after = [keyframe(32, 16), keyframe(32, 16)], [keyframe(16, 32), keyframe(16, 32)]
    assert ingest.reconcile_orientation(before, after) == 16
    for kf in before + after:
        assert kf.image.shape == (16, 16, 3) and kf.image.flags.c_contiguous
        assert kf.image[8, 8, 0] == 255  # the centre is kept
    assert ingest.reconcile_orientation([keyframe(16, 32)], [keyframe(16, 32)]) == 0
    assert ingest.reconcile_orientation([keyframe(16, 16)], [keyframe(16, 32)]) == 0  # square
    assert ingest.reconcile_orientation([], after) == 0


def test_ingest_pair_mixed_orientation_writes_squares(tmp_path, pan_then_still_video):
    frames = [np.full((64, 48, 3), 40 + 3 * k, np.uint8) for k in range(60)]  # portrait
    portrait = write_video(tmp_path / "after.mp4", frames)  # the fixture video is landscape
    cfg = load_config(overrides=["ingest.num_keyframes=4", "ingest.min_motion=0"])
    index = ingest_pair(pan_then_still_video, portrait, tmp_path / "out", cfg)
    assert (index["before"]["square_crop"], index["after"]["square_crop"]) == (96, 48)
    for name, side in (("before", 96), ("after", 48)):
        sizes = {Image.open(p).size for p in (tmp_path / "out" / "keyframes" / name).glob("*.jpg")}
        assert sizes == {(side, side)}
