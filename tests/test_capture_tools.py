"""Tests for the capture-day scripts: check_video.py, score_gt.py and lidar_check.py."""

import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType

import av
import cv2
import numpy as np
import pytest
import yaml
from plyfile import PlyData, PlyElement

from roomdiff.config import load_config
from roomdiff.geometry import rotation_y

ROOT = Path(__file__).resolve().parents[1]
OFFICE_PLY = ROOT / "outputs" / "scenediff_office_1" / "viewer" / "before.ply"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_video = load_script("check_video")
score_gt = load_script("score_gt")
lidar_check = load_script("lidar_check")

# ------------------------------------------------------------------- check_video


def write_video(path: Path, seconds: float, speed: int = 1, size: tuple[int, int] = (320, 240),
                rotation: int | None = None, seed: int = 0) -> str:
    """H.264 video (PyAV) panning over a strip of ~128 px black / white blobs with fine
    noise, `speed` px per frame (0 = a still camera); `rotation` = display matrix angle."""
    w, h = size
    n = round(seconds * 30)
    length = w + speed * n
    rng = np.random.default_rng(seed)
    coarse = rng.random((h // 128 + 2, length // 128 + 2))
    field = cv2.resize(coarse, (length + 256, h + 256), interpolation=cv2.INTER_CUBIC)
    blobs = np.where(field[:h, :length] > np.median(field), 220.0, 40.0)[..., None]
    fine = rng.integers(-25, 26, (h // 4, length // 4 + 1, 3)).repeat(4, 0).repeat(4, 1)
    strip = np.clip(blobs + fine[:h, :length], 0, 255).astype(np.uint8)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        if rotation is not None:
            stream.set_display_rotation(rotation)
        for i in range(n):
            image = np.ascontiguousarray(strip[:, speed * i: speed * i + w])
            container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
        container.mux(stream.encode())
    return str(path)


def statuses(result: dict) -> dict[str, str]:
    return {c["check"]: c["status"] for c in result["checks"]}


@pytest.fixture(scope="module")
def ingest_params() -> dict:
    return dict(load_config()["ingest"])


def test_check_video_good_pair_passes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    before = write_video(tmp_path / "before.mp4", 31, seed=1)
    after = write_video(tmp_path / "after.mov", 31, seed=2)
    out = tmp_path / "check.json"
    assert check_video.main([before, after, "--json", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["result"] == "PASS", capsys.readouterr().out
    video = report["videos"]["before"]
    assert video["keyframes"]["usable"] == 48
    assert 55 <= len(video["samples"]["sharpness"]) <= 60
    assert report["pair"][0]["status"] == "PASS"
    assert "RESULT: PASS" in capsys.readouterr().out


def test_check_video_hard_failures(tmp_path: Path) -> None:
    # 5 s, still camera, portrait (display matrix -90 like an iPhone) vs a landscape video
    before = write_video(tmp_path / "before.mov", 5, speed=0, rotation=-90)
    after = write_video(tmp_path / "after.mp4", 16)
    out = tmp_path / "check.json"
    assert check_video.main([before, after, "--json", str(out)]) == 1
    report = json.loads(out.read_text())
    first = statuses(report["videos"]["before"])
    assert first["duration"] == "FAIL"
    assert first["orientation"] == "WARN"  # portrait
    assert report["videos"]["before"]["info"]["width"] == 240  # upright after the rotation
    assert first["keyframes"] == "FAIL"  # camera at rest: every pick but the first
    assert report["videos"]["before"]["keyframes"]["usable"] == 1
    assert statuses(report["videos"]["after"])["duration"] == "WARN"  # 16 s
    assert report["pair"][0]["status"] == "FAIL"  # mixed orientation
    assert report["result"] == "FAIL"


def test_check_video_fast_pan_and_hdr_warn(tmp_path: Path, ingest_params: dict) -> None:
    fast = check_video.check_video(write_video(tmp_path / "fast.mp4", 16, speed=24),
                                   ingest_params)
    assert statuses(fast)["motion"] == "WARN"
    assert fast["fast_fraction"] > 0.5
    hdr = tmp_path / "hdr.mov"  # 10-bit HEVC with HLG transfer and BT.2020 primaries
    with av.open(str(hdr), "w") as container:
        stream = container.add_stream("libx265", rate=30, options={"x265-params": "log-level=none"})
        stream.width, stream.height, stream.pix_fmt = 160, 128, "yuv420p10le"
        stream.codec_context.color_trc, stream.codec_context.color_primaries = 18, 9
        for _ in range(30):
            frame = av.VideoFrame.from_ndarray(np.full((128, 160, 3), 90, np.uint8), "rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    codec = check_video.codec_info(str(hdr))
    assert (codec["codec"], codec["bits"], codec["transfer"], codec["bt2020"]) == (
        "hevc", 10, "HLG", True)
    result = check_video.check_video(str(hdr), ingest_params)
    message = next(c["message"] for c in result["checks"] if c["check"] == "codec")
    assert statuses(result)["codec"] == "WARN" and "HDR" in message


def test_check_video_unreadable(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mov"
    junk.write_bytes(b"not a video" * 100)
    good = write_video(tmp_path / "good.mp4", 16)
    out = tmp_path / "check.json"
    assert check_video.main([str(junk), good, "--json", str(out)]) == 1
    report = json.loads(out.read_text())
    assert statuses(report["videos"]["before"]) == {"readable": "FAIL"}
    assert report["pair"] == []


# ---------------------------------------------------------------------- score_gt


@pytest.mark.parametrize(("a", "b", "match"), [
    ("red office chair", "office chair", True),
    ("office chair", "chair", True),
    ("office chair", "desk chair", True),
    ("seat", "grey chair", True),
    ("backpack", "black bag", True),
    ("box", "cardboard carton", True),
    ("couch", "sofa", True),
    ("jumper", "wool sweater", True),
    ("trash can", "bin", True),
    ("pyjamas", "pajamas", True),
    ("lamp on the desk", "desk lamp", True),
    ("bookshelves", "bookshelf", True),
    ("coffee table", "coffee mug", False),
    ("chair cushion", "chair", False),
    ("book", "notebook", False),
    ("object", "chair", False),
])
def test_label_match(a: str, b: str, match: bool) -> None:
    score = score_gt.label_score(score_gt.label_tokens(a), score_gt.label_tokens(b))
    assert (score >= 0.6) == match, (a, b, score)


def change(cid: int, kind: str, label: str, conf: float, **extra: object) -> dict:
    return {"id": cid, "type": kind, "label": label, "confidence": conf, "before": None,
            "after": None, "transform": None, **extra}


SIDE = {"centroid": [0.0, 0.5, 0.0], "obb": {"center": [0, 0.5, 0], "extents": [0.5, 1, 0.5],
                                             "yaw_deg": 0.0}}
PREDICTIONS = [
    change(1, "moved", "office chair", 0.9, before=SIDE, after=SIDE, transform={
        "translation_m": 1.0, "rotation_deg": 80.0, "rotation_reliable": True}),
    change(2, "removed", "pillow", 0.8, before=SIDE),
    change(7, "removed", "cushion", 0.6, before=SIDE),
    change(3, "added", "black bag", 0.7, after=SIDE),
    change(4, "removed", "desk lamp", 0.65, before={"centroid": [0.0, 0.4, 0.0]}),
    change(5, "added", "lamp", 0.6, after={"centroid": [0.6, 0.4, 0.8]}),
    change(6, "removed", "book", 0.5, before=SIDE),
    change(8, "added", "towel", 0.5, after=SIDE),
    change(9, "unverified", "wall", 0.3, after=SIDE),
    change(10, "unverified", "trash can", 0.4, after=SIDE),
    change(11, "removed", "cup", 0.55, before=SIDE),
]
GT = {
    "scene": "synthetic",
    "changes": [
        {"object": "red office chair", "type": "moved", "displacement_m": 1.1, "rotation_deg": 90},
        {"object": "cushion", "type": "removed"},
        {"object": "backpack", "type": "added", "size_m": [0.3, 0.2, 0.45]},
        {"object": "lamp", "type": "moved", "displacement_m": 0.8, "where": "bedside"},
        {"object": "book", "type": "moved"},
        {"object": "plant", "type": "added"},
    ],
    "unchanged_but_only_in_one_video": [
        {"object": "lamp on the bedside table", "seen_in": "before"},
        {"object": "bin", "seen_in": "after"},
        {"object": "mug", "seen_in": "before"},
    ],
}


def test_score_gt_matching() -> None:
    report = score_gt.score(GT, PREDICTIONS)
    rows = {r["object"]: r for r in report["gt"]}
    chair = rows["red office chair"]
    assert (chair["status"], chair["predictions"][0]["id"]) == ("found", 1)
    assert chair["displacement_err_m"] == pytest.approx(-0.1)
    assert chair["rotation_err_deg"] == pytest.approx(10.0)
    assert rows["cushion"]["predictions"][0]["id"] == 2  # most confident synonym wins
    assert rows["backpack"]["predictions"][0]["id"] == 3
    lamp = rows["lamp"]  # moved object reported as removed + added
    assert lamp["status"] == "type wrong" and not lamp["type_correct"]
    assert [p["id"] for p in lamp["predictions"]] == [4, 5]
    assert lamp["displacement_pred_m"] == pytest.approx(1.0)  # centroid distance
    assert lamp["rotation_pred_deg"] is None
    assert rows["book"]["status"] == "partial"
    assert rows["plant"]["status"] == "missed"
    assert report["counts"] == {"gt": 6, "found": 4, "type_correct": 3, "missed": 2,
                                "false_positives": 2, "unchanged_correct": 2, "unchanged": 3}
    assert {e["id"] for e in report["extra"]} == {7, 8}
    assert [e["id"] for e in report["extra_unverified"]] == [9]
    unchanged = {u["object"]: u for u in report["unchanged"]}
    assert unchanged["lamp on the bedside table"]["correct"]  # lamp halves already used
    assert unchanged["bin"]["correct"] and unchanged["bin"]["unverified"][0]["id"] == 10
    assert not unchanged["mug"]["correct"]
    assert unchanged["mug"]["reported_as_change"][0]["id"] == 11
    scaled = score_gt.score(GT, PREDICTIONS, scale=1.1)["gt"][0]
    assert scaled["displacement_err_m"] == pytest.approx(0.0)


def test_gt_template_is_valid() -> None:
    gt = score_gt.load_gt(ROOT / "docs" / "gt_template.yaml")
    assert [c["type"] for c in gt["changes"]] == ["moved", "added"]
    assert gt["unchanged_but_only_in_one_video"][0]["seen_in"] == "before"


def test_score_gt_cli(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / "changes.json").write_text(json.dumps({"changes": PREDICTIONS}))
    gt = tmp_path / "gt.yaml"
    gt.write_text(yaml.safe_dump(GT))
    assert score_gt.main([str(tmp_path), str(gt), "--json"]) == 0
    text = capsys.readouterr().out
    assert "| 4 | lamp | moved | type wrong | #4 removed + #5 added |" in text
    assert "WRONG: reported as removed #11 (cup)" in text
    saved = json.loads((tmp_path / "gt_score.json").read_text())
    assert saved["counts"]["found"] == 4
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"changes": [
        {"object": "chair", "type": "moved", "displacment_m": 1.0},
        {"object": "box", "type": "added", "displacement_m": 1.0}]}))
    assert score_gt.main([str(tmp_path), str(bad)]) == 2
    err = capsys.readouterr().err
    assert "unknown key 'displacment_m'" in err and "for moved objects only" in err


# ------------------------------------------------------------------- lidar_check


def l_room(rng: np.random.Generator, spacing: float = 0.02) -> np.ndarray:
    """Random points on the floor, the 2.5 m walls and three boxes of an L-shaped room."""
    def patch(origin: list[float], u: list[float], v: list[float]) -> np.ndarray:
        u_, v_ = np.asarray(u, float), np.asarray(v, float)
        n = int(np.linalg.norm(u_) * np.linalg.norm(v_) / spacing**2)
        ab = rng.random((n, 2))
        return np.asarray(origin, float) + ab[:, :1] * u_ + ab[:, 1:] * v_

    corners = [(0, 0), (5, 0), (5, 2.5), (3, 2.5), (3, 4), (0, 4)]  # (x, z)
    parts = [patch([0, 0, 0], [5, 0, 0], [0, 0, 2.5]), patch([0, 0, 2.5], [3, 0, 0], [0, 0, 1.5])]
    for (x0, z0), (x1, z1) in zip(corners, corners[1:] + corners[:1]):
        parts.append(patch([x0, 0, z0], [x1 - x0, 0, z1 - z0], [0, 2.5, 0]))
    for x, z, w, h, d in [(0.3, 0.3, 1.2, 0.75, 0.8), (4.0, 0.2, 0.8, 1.8, 0.4),
                          (0.4, 2.8, 2.0, 0.5, 1.0)]:
        parts += [patch([x, h, z], [w, 0, 0], [0, 0, d]),  # top and four sides
                  patch([x, 0, z], [w, 0, 0], [0, h, 0]),
                  patch([x, 0, z + d], [w, 0, 0], [0, h, 0]),
                  patch([x, 0, z], [0, 0, d], [0, h, 0]),
                  patch([x + w, 0, z], [0, 0, d], [0, h, 0])]
    return np.concatenate(parts)


def write_ply(path: Path, points: np.ndarray) -> None:
    vertex = np.empty(len(points), [("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    vertex["x"], vertex["y"], vertex["z"] = points.T
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))


def fake_scan(points: np.ndarray, scale: float, yaw: float, t: list[float],
              rng: np.random.Generator) -> np.ndarray:
    """The cloud scaled, turned about +Y, moved, with 1 cm noise and 30 % of it dropped."""
    scan = scale * points @ rotation_y(yaw).T + np.asarray(t)
    scan += rng.normal(0.0, 0.01, scan.shape)
    return scan[rng.random(len(scan)) >= 0.3]


@pytest.mark.parametrize("source", ["office_1", "l_room"])
def test_lidar_check_recovers_scale_and_yaw(tmp_path: Path, source: str,
                                            capsys: pytest.CaptureFixture) -> None:
    rng = np.random.default_rng(0)
    if source == "office_1":
        if not OFFICE_PLY.is_file():
            pytest.skip("outputs/scenediff_office_1 is not available")
        ours = {"before": lidar_check.read_ply(OFFICE_PLY)}
    else:  # also the after recording: a second scan session with its own yaw and origin
        ours = {"before": l_room(rng), "after": l_room(rng)}
    poses = {"before": (35.0, [2.0, -1.3, -0.7]), "after": (-120.0, [-1.0, -1.25, 3.0])}
    args = [str(tmp_path), "--json"]
    for rec, points in ours.items():
        write_ply(tmp_path / "viewer" / f"{rec}.ply", points)
        write_ply(tmp_path / "lidar" / f"{rec}.ply", fake_scan(points, 1.07, *poses[rec], rng))
        args += [f"--{rec}", str(tmp_path / "lidar" / f"{rec}.ply")]
    moved = change(1, "moved", "chair", 0.9, before=SIDE, after=SIDE,
                   transform={"translation_m": 1.0})
    (tmp_path / "changes.json").write_text(json.dumps({"changes": [moved]}))

    start = time.perf_counter()
    assert lidar_check.main(args) == 0
    assert time.perf_counter() - start < 60
    report = json.loads((tmp_path / "lidar_check.json").read_text())
    for rec, fit in report["fits"].items():
        assert fit["scale"] == pytest.approx(1.07, rel=0.01), rec
        assert abs((fit["yaw_deg"] - poses[rec][0] + 180) % 360 - 180) < 2, rec
        assert fit["overlap_10cm"] > 0.9 and fit["residual_median_m"] < 0.02
    row = report["changes"][0]
    assert row["distance_metric_m"] == pytest.approx(report["scale"], abs=1e-3)
    assert row["size_metric_m"][1] == pytest.approx(report["scale"], abs=1e-3)
    if len(ours) == 2:
        assert report["consistency"]["status"] == "PASS"
    print(capsys.readouterr().out)  # shown with pytest -s


def test_floor_height_prefers_floor_to_a_bed() -> None:
    rng = np.random.default_rng(0)
    floor = rng.normal(-1.4, 0.02, 20_000)  # tilted / noisy: smeared over several cm
    bed = rng.normal(-0.95, 0.005, 40_000)  # a larger, flatter layer 45 cm higher
    assert lidar_check.floor_height(np.concatenate([floor, bed])) == pytest.approx(-1.4, abs=0.03)


def test_our_cloud_falls_back_to_recon_npz(tmp_path: Path) -> None:
    (tmp_path / "recon").mkdir()
    depth = np.full((3, 4, 5), 2.0, np.float32)
    valid = np.ones((3, 4, 5), bool)
    valid[2, 0] = False
    K = np.tile(np.array([[5.0, 0, 2], [0, 5.0, 1.5], [0, 0, 1]], np.float32), (3, 1, 1))
    np.savez(tmp_path / "recon" / "recon.npz", rec=np.array([0, 1, 1], np.uint8), valid=valid,
             depth=depth, K=K, cam2world=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)))
    points, source = lidar_check.our_cloud(tmp_path, "after")
    assert points.shape == (2 * 20 - 5, 3) and "recon.npz" in source
    assert np.allclose(points[:, 2], 2.0)
