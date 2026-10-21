"""Tests for the appearance channel (CPU, fake models, synthetic scenes).

Scene (OpenCV world, y down): a uniformly blue wall at z = 2 m filling every view; three
before cameras at z = 0 and three after cameras at z = 0.1 m, all looking along +z. The
in-place change paints a 0.8 m square of the wall red in the after recording only (the
geometry is identical); the lighting change darkens every after image instead. The fake
featurizer's tokens are the mean colour of each 16 px patch.
"""

import json

import numpy as np
import pytest
from PIL import Image

from roomdiff.appearance import (
    Candidate,
    Region,
    aggregate,
    consistent_share,
    covisibility,
    is_static,
    max_entropy_threshold,
    near_fraction,
    pair_maps,
    patch_centres,
    run_appearance,
    sample_tokens,
    select_partners,
    stride_points,
)
from roomdiff.changes import replaced_pairs, run_changes
from roomdiff.config import load_config
from roomdiff.geometry import obb_overlap
from roomdiff.instances import load_instances, run_instances
from roomdiff.models import FakeFeaturizer, FakeSegmenter
from roomdiff.recon import write_recon
from roomdiff.semantics import run_semantics
from roomdiff.visibility import run_visibility
from roomdiff.vlm import FakeVLM

H, W = 240, 320
K = np.array([[160.0, 0.0, 159.5], [0.0, 160.0, 119.5], [0.0, 0.0, 1.0]], np.float32)
WALL_Z = 2.0
WALL = (20, 20, 220)
RED = (220, 20, 20)
BLOCK_LO, BLOCK_HI = np.array([-0.4, -0.4]), np.array([0.4, 0.4])  # x, y on the wall
CAMS = {"before": [(-0.3, 0.0, 0.0), (0.0, 0.05, 0.0), (0.3, 0.0, 0.0)],
        "after": [(-0.25, 0.0, 0.1), (0.05, -0.05, 0.1), (0.35, 0.0, 0.1)]}
SETTINGS = ["visibility.device=cpu", "visibility.voxel_size=0.02", "instances.device=cpu",
            "appearance.score_threshold=0.5"]
PARAMS = dict(load_config().appearance)


def pose(c) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = c
    return T


def render(c, boxes=()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z-depth, world hit points (H, W, 3) and box ids (0 = wall) seen from camera centre c
    looking along +z; `boxes` = [(lo, hi), ...] axis-aligned boxes in front of the wall."""
    c = np.asarray(c, np.float64)
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1)
    depth = np.full((H, W), WALL_Z - c[2])
    ids = np.zeros((H, W), np.int32)
    for k, (lo, hi) in enumerate(boxes, start=1):
        with np.errstate(divide="ignore", invalid="ignore"):
            t0, t1 = (np.asarray(lo) - c) / d, (np.asarray(hi) - c) / d
        near, far = np.minimum(t0, t1).max(-1), np.maximum(t0, t1).min(-1)
        hit = (near < far) & (near > 0) & (near < depth)
        depth, ids = np.where(hit, near, depth), np.where(hit, k, ids)
    return depth.astype(np.float32), c + d * depth[..., None], ids


def make_recon(paint: bool = True, light: float = 1.0) -> tuple[dict, dict[str, np.ndarray]]:
    """Recon dict of the wall scene and SAM label maps (1 = the painted block)."""
    data = {k: [] for k in ("rec", "files", "images", "cam2world", "K", "depth", "conf",
                            "valid")}
    labels = {}
    for rec_id, name in enumerate(("before", "after")):
        for j, c in enumerate(CAMS[name]):
            depth, hit, _ = render(c)
            block = np.all((hit[..., :2] >= BLOCK_LO) & (hit[..., :2] <= BLOCK_HI), -1)
            image = np.empty((H, W, 3), np.float64)
            image[:] = WALL
            if paint and name == "after":
                image[block] = RED
            if name == "after":
                image *= light
            labels[str(len(data["rec"]))] = block.astype(np.int32)
            data["rec"].append(rec_id)
            data["files"].append(f"{name}_{j:05d}.jpg")
            data["images"].append(np.round(image).astype(np.uint8))
            data["cam2world"].append(pose(c))
            data["K"].append(K)
            data["depth"].append(depth)
            data["conf"].append(np.full((H, W), 0.9, np.float32))
            data["valid"].append(np.ones((H, W), bool))
    recon = {k: np.stack(v) for k, v in data.items()}
    recon["metric_scale"] = 1.0
    return recon, labels


def write_scene(out, **kwargs) -> dict[str, np.ndarray]:
    recon, labels = make_recon(**kwargs)
    (out / "recon").mkdir(parents=True)
    write_recon(out / "recon" / "recon.npz", recon)
    (out / "recon" / "meta.json").write_text(json.dumps({"diagnostics": {}}))
    return labels


# ------------------------------------------------------------------ cue maps


def test_covisibility_of_a_synthetic_pair():
    recon, _ = make_recon()
    # same wall, cameras 0.3 m apart: most of each view lies in the other one
    share = consistent_share(stride_points(recon, 0, 8), recon, 3, 0.03, 0.05)
    assert 0.8 < share <= 1.0
    assert consistent_share(stride_points(recon, 0, 8), recon, 0, 0.03, 0.05) == 1.0
    # a camera far to the side sees none of it; one looking away neither
    side = {**recon, "cam2world": recon["cam2world"].copy()}
    side["cam2world"][3, :3, 3] = [6.0, 0.0, 0.1]
    assert consistent_share(stride_points(side, 0, 8), side, 3, 0.03, 0.05) == 0.0
    turned = {**recon, "cam2world": recon["cam2world"].copy()}
    turned["cam2world"][3, :3, :3] = np.diag([-1.0, 1.0, -1.0])
    assert consistent_share(stride_points(turned, 0, 8), turned, 3, 0.03, 0.05) == 0.0
    # wrong depth (the other camera sees a surface 0.5 m nearer): not depth-consistent
    near = {**recon, "depth": recon["depth"].copy()}
    near["depth"][3] -= 0.5
    assert consistent_share(stride_points(near, 0, 8), near, 3, 0.03, 0.05) == 0.0
    scores = covisibility(recon, [0, 1, 2], [3, 4, 5], 8, 0.03, 0.05)
    assert scores.shape == (3, 3) and np.all(scores > 0.6)
    assert scores[0, 0] > scores[0, 2]  # the left cameras overlap more with each other
    assert select_partners(scores, 2, 0.3)[0][0] == 0
    assert select_partners(np.array([[0.9, 0.2, 0.5]]), 2, 0.3) == [[0, 2]]
    assert select_partners(np.array([[0.1, 0.2]]), 2, 0.3) == [[]]


def test_token_sampling_and_patch_centres():
    rows, cols = patch_centres(15, 20, H, W)
    assert rows.shape == (15, 20) and rows[0, 0] == 8 and cols[0, -1] == W - 8
    tokens = np.zeros((2, 2, 2), np.float32)
    tokens[:, 0] = [1, 0]
    tokens[:, 1] = [0, 1]
    left, mid = sample_tokens(tokens, np.array([0.0, 99.5]), np.array([50.0, 50.0]), 100, 200)
    assert np.allclose(left, [1, 0]) and np.allclose(mid, [np.sqrt(0.5)] * 2, atol=1e-6)


def test_geometric_cue_fires_only_where_the_surface_vanished():
    """A box stands in front of the wall before and is gone after. From the before frame
    the after camera sees through the box surface (geom); from the after frame the wall
    where the box stood is hidden behind the box in the before frame: no vote at all."""
    box = (np.array([-0.2, -0.2, 1.2]), np.array([0.2, 0.2, 1.5]))
    data = {k: [] for k in ("depth", "valid", "K", "cam2world")}
    ids = []
    for c, boxes in (((0.0, 0.0, 0.0), [box]), ((0.1, 0.0, 0.0), [])):
        depth, _, i = render(c, boxes)
        data["depth"].append(depth)
        data["valid"].append(np.ones((H, W), bool))
        data["K"].append(K)
        data["cam2world"].append(pose(c))
        ids.append(i)
    recon = {k: np.stack(v) for k, v in data.items()}
    tokens = np.zeros((15, 20, 4), np.float32)
    tokens[..., 0] = 1.0  # identical features everywhere: only the depth cue can fire
    rows, cols = patch_centres(15, 20, H, W)
    on_box = ids[0][rows, cols] == 1
    forward = pair_maps(recon, 0, 1, tokens, tokens, PARAMS)
    assert on_box.sum() > 10
    assert forward.geom[on_box].mean() > 0.9 and not forward.geom[~on_box].any()
    assert forward.observed[on_box].all() and np.allclose(forward.feat, 0, atol=1e-6)
    backward = pair_maps(recon, 1, 0, tokens, tokens, PARAMS)
    assert not backward.geom.any()
    behind = (rows >= 105) & (rows <= 135) & (cols >= 140) & (cols <= 180)  # hidden wall
    assert behind.any() and not backward.observed[behind].any()
    score = aggregate(0, [forward, forward], (15, 20), 1.0, 2)
    assert np.allclose(score.score[on_box & forward.geom], 1.0)
    assert np.all(aggregate(0, [forward], (15, 20), 1.0, 2).score == 0)  # one vote < min_votes


def test_max_entropy_threshold_splits_two_modes():
    rng = np.random.default_rng(0)
    low, high = rng.normal(0.2, 0.03, 5000), rng.normal(0.9, 0.03, 300)
    t = max_entropy_threshold(np.concatenate([low, high]))
    # Kapur's threshold sits in the upper tail of the dominant mode, below the small one
    assert t is not None and (low > t).mean() < 0.05 and (high > t).all()
    assert max_entropy_threshold(np.zeros(10)) is None


# ------------------------------------------------------------------ static filter


def candidate(static_fraction: float, feat: float, geom: float) -> Candidate:
    mask = np.zeros((H, W), bool)
    mask[10:20, 10:20] = True
    region = Region(0, 0, mask, (10, 10, 19, 19), 100, feat + geom, feat, geom,
                    np.array([[15.0, 15.0]]), [3])
    return Candidate(region, mask, 0.9, 1.0, np.zeros((50, 3), np.float32),
                     np.zeros((50, 3), np.uint8), static_fraction, (10, 10, 19, 19))


def test_static_surface_filter():
    from scipy.spatial import cKDTree

    wall = np.stack(np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01), [2.0],
                                indexing="ij"), -1).reshape(-1, 3)
    flat = wall[:400] + [0, 0, -0.01]  # 1 cm in front of the wall: lies on it
    raised = wall[:400] + [0, 0, -0.10]
    tree = cKDTree(wall)
    assert near_fraction(flat, tree, 0.02) == 1.0 and near_fraction(raised, tree, 0.02) == 0.0
    assert near_fraction(flat, None, 0.02) == 0.0
    assert is_static(candidate(0.9, feat=0.1, geom=0.6), 0.5)  # depth noise on the wall
    assert not is_static(candidate(0.9, feat=0.7, geom=0.1), 0.5)  # flat object / repaint
    assert not is_static(candidate(0.3, feat=0.1, geom=0.9), 0.5)  # off the surfaces


# ------------------------------------------------------------------ stage


def test_in_place_change_yields_one_candidate_per_recording(tmp_path):
    labels = write_scene(tmp_path)
    cfg = load_config(overrides=SETTINGS)
    meta = run_appearance(tmp_path, cfg, FakeSegmenter(labels), FakeFeaturizer())
    assert meta["clusters"] == {"before": 1, "after": 1}
    doc = json.loads((tmp_path / "appearance" / "clusters.json").read_text())
    for c in doc["clusters"]:
        assert np.allclose(c["centroid"], [0.0, 0.0, WALL_Z], atol=0.05)
        assert max(c["obb"]["extents"]) == pytest.approx(0.8, abs=0.06)
        ev = c["evidence"]
        assert ev["frames_supporting"] == 3 and ev["feat_term"] > 0.5 and ev["geom_term"] == 0
        assert ev["static_fraction"] > 0.9  # lies on the shared wall: kept as feature-driven
        assert ev["partners_visible"] >= 2 and ev["observing_views"] == 3
    assert meta["threshold"]["max_entropy"] is not None
    assert (tmp_path / "appearance" / "feats_before.npy").is_file()
    tokens = np.load(tmp_path / "appearance" / "feats_after.npy")
    assert tokens.dtype == np.float16 and tokens.shape == (3, 15, 20, FakeFeaturizer.dim)
    again = run_appearance(tmp_path, cfg, FakeSegmenter(labels), FakeFeaturizer())
    assert again["features_cached"] == {"before": True, "after": True}
    assert again["clusters"] == meta["clusters"]


def test_lighting_only_change_yields_no_candidate(tmp_path):
    labels = write_scene(tmp_path, paint=False, light=0.6)
    meta = run_appearance(tmp_path, load_config(overrides=SETTINGS), FakeSegmenter(labels),
                          FakeFeaturizer())
    assert meta["regions"] == {"before": 0, "after": 0}
    assert meta["clusters"] == {"before": 0, "after": 0}
    assert meta["score_stats"]["p95"] < 0.05


def test_disabled_stage_writes_nothing(tmp_path):
    write_scene(tmp_path)
    cfg = load_config(overrides=[*SETTINGS, "appearance.enabled=false"])
    assert run_appearance(tmp_path, cfg, FakeSegmenter(), FakeFeaturizer()) == {"enabled": False}
    assert not (tmp_path / "appearance").exists()


@pytest.fixture(scope="module")
def replaced_scene(tmp_path_factory):
    out = tmp_path_factory.mktemp("appearance")
    labels = write_scene(out)
    cfg = load_config(overrides=SETTINGS)
    run_visibility(out, cfg)
    run_appearance(out, cfg, FakeSegmenter(labels), FakeFeaturizer())
    meta = run_instances(out, cfg, FakeSegmenter(labels), FakeFeaturizer())
    doc = run_changes(out, cfg)
    return out, meta, doc


def test_replaced_pair_is_linked(replaced_scene):
    out, meta, doc = replaced_scene
    assert meta["sources"] == {"before": {"geometry": 0, "appearance": 1, "detection": 0},
                               "after": {"geometry": 0, "appearance": 1, "detection": 0}}
    assert len(meta["appearance"]["added"]) == 2 and meta["appearance"]["covered"] == []
    for name in ("before", "after"):
        data = load_instances(out, name)
        (record,) = data["instances"]
        assert record["source"] == "appearance" and record["blob_ids"] == []
        assert record["kind"] == ("disappeared" if name == "before" else "appeared")
        assert record["evidence"]["appearance"]["frames_supporting"] == 3
        assert record["n_frame_masks"] == 3 and data["feats"].shape == (1, FakeFeaturizer.dim)
    types = sorted(c["type"] for c in doc["changes"])
    assert types == ["added", "removed"]  # never a 0 m "moved" pair
    removed = next(c for c in doc["changes"] if c["type"] == "removed")
    added = next(c for c in doc["changes"] if c["type"] == "added")
    assert removed["replaced_by"] == added["id"] and added["replaces"] == removed["id"]
    assert removed["evidence"]["replace_overlap"] > 0.9
    matching = json.loads((out / "instances" / "matching.json").read_text())
    assert matching["replaced"] == [[1, 1, removed["evidence"]["replace_overlap"]]]


def test_changes_json_schema_with_appearance_changes(replaced_scene):
    out, _, doc = replaced_scene
    assert json.loads((out / "changes.json").read_text()) == doc
    ids = {c["id"]: c for c in doc["changes"]}
    for c in doc["changes"]:
        for key, kind in {"id": int, "type": str, "label": str, "confidence": float,
                          "evidence": dict, "description": str, "crops": dict}.items():
            assert isinstance(c[key], kind), key
        assert 0.0 <= c["confidence"] <= 1.0 and c["transform"] is None
        side = c["before"] if c["type"] == "removed" else c["after"]
        assert c["after" if c["type"] == "removed" else "before"] is None
        assert side["source"] == "appearance" and side["blob_ids"] == []
        assert set(side["obb"]) == {"center", "extents", "yaw_deg"}
        for key in ("free_rays", "free_frames_mean", "observing_views", "occ_points",
                    "mask_score", "view_consistency", "vlm_verdict"):
            assert key in c["evidence"], key
        app = c["evidence"]["appearance"]
        assert set(app) == {"cluster", "frames_supporting", "mean_score", "feat_term",
                            "geom_term", "partners_visible"}
        link = c.get("replaced_by", c.get("replaces"))
        assert isinstance(link, int) and ids[link]["type"] != c["type"]
        assert all(path is None or (out / path).is_file() for path in c["crops"].values())


def test_viz_logs_appearance(replaced_scene):
    from roomdiff.viz_rerun import heat_overlay, log_scene

    out, _, _ = replaced_scene
    assert log_scene(out, max_points=5000).stat().st_size > 10_000
    heat = heat_overlay(np.zeros((4, 4, 3), np.uint8), np.array([[0, 0.5, 1, 2]] * 4))
    assert heat.shape == (4, 4, 3) and heat[0, 0].sum() == 0 and tuple(heat[0, 3]) == (127,) * 3


def test_replaced_pairs_by_box_overlap():
    def rec(center, extents, source="appearance"):
        return {"obb": {"center": center, "extents": extents, "yaw_deg": 0.0},
                "source": source}

    wallet = rec([0, 0.71, 0], [0.12, 0.02, 0.09])  # on a table
    table_patch = rec([0.01, 0.70, 0.0], [0.14, 0.005, 0.1])  # the bare table where it lay
    far = rec([1.0, 0.7, 0], [0.12, 0.02, 0.09])
    assert obb_overlap(wallet["obb"], table_patch["obb"], 0.02) > 0.5
    pairs = replaced_pairs([wallet, far], [far, table_patch], 0.3)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (1, 0)]
    assert replaced_pairs([wallet], [rec([0.5, 0.7, 0], [0.1, 0.02, 0.1])], 0.3) == []


# ------------------------------------------------------------------ semantics


COLOURS = {"red": (210, 30, 30), "blue": (30, 30, 210), "grey": (128, 128, 128)}


def colour(image) -> str:
    r, _, b = np.asarray(image, float).reshape(-1, 3).mean(0)
    return "red" if r > 150 else "blue" if b > 150 else "grey"


def side(iid, centroid):
    c = np.asarray(centroid, float)
    return {"instance_id": iid, "centroid": c.tolist(), "aabb": [(c - 0.1).tolist(),
            (c + 0.1).tolist()], "extents": [0.2] * 3,
            "obb": {"center": c.tolist(), "extents": [0.2] * 3, "yaw_deg": 0.0},
            "n_points": 300, "blob_ids": [], "source": "appearance"}


def write_pair(root, before_colour, after_colour, label_before="wallet",
               label_after="phone"):
    """A linked removed (1) / added (2) pair at one place and an unlinked removed
    appearance change (3) elsewhere whose place view shows the same blue thing."""
    (root / "crops").mkdir(exist_ok=True)

    def crop(name, col):
        Image.new("RGB", (400, 300), COLOURS[col]).save(root / "crops" / f"{name}.jpg")
        return f"crops/{name}.jpg"

    ev = {"free_rays": 0, "free_frames_mean": 0.0, "observing_views": 5, "occ_points": 300,
          "mask_score": 0.9, "view_consistency": 0.8, "vlm_verdict": None,
          "appearance": {"cluster": 1, "frames_supporting": 3, "mean_score": 0.8,
                         "feat_term": 0.8, "geom_term": 0.0, "partners_visible": 2}}
    changes = [
        {"id": 1, "type": "removed", "label": label_before, "confidence": 0.9,
         "before": side(1, [0, 0, 2]), "after": None, "transform": None, "evidence": dict(ev),
         "description": "", "crops": {"before": crop("1_before", before_colour),
                                      "after": crop("1_after", after_colour)},
         "replaced_by": 2},
        {"id": 2, "type": "added", "label": label_after, "confidence": 0.8, "before": None,
         "after": side(1, [0, 0, 2]), "transform": None, "evidence": dict(ev),
         "description": "", "crops": {"after": crop("2_after", after_colour),
                                      "before": crop("2_before", before_colour)},
         "replaces": 1},
        {"id": 3, "type": "removed", "label": "shirt", "confidence": 0.85,
         "before": side(2, [2, 0, 2]), "after": None, "transform": None, "evidence": dict(ev),
         "description": "", "crops": {"before": crop("3_before", "blue"),
                                      "after": crop("3_after", "blue")}},
    ]
    doc = {"scene": {"units": "metres (approx.)"}, "changes": changes, "summary": ""}
    (root / "changes.json").write_text(json.dumps(doc))


def in_place_vlm(answers):
    """Scripted VLM: names crops by colour, answers the in-place question from the two
    crop colours via `answers[(c1, c2)]`."""
    def respond(images, prompt):
        cols = [colour(im) for im in images]
        if '"label"' in prompt and "What changed" not in prompt:
            return json.dumps({"label": {"red": "wallet", "blue": "phone",
                                         "grey": "bed sheet"}[cols[0]],
                               "attributes": cols[0], "description": "x"})
        if "same physical object" in prompt:  # re-pairing: never the same object here
            return '{"verdict": "rejected", "reason": "different"}'
        if "What changed at this place" in prompt:
            return json.dumps({"reason": "scripted", "change": answers[tuple(cols)]})
        if '"summary"' in prompt:
            return '{"summary": "ok"}'
        return '{"verdict": "confirmed", "reason": "fallback"}'
    return FakeVLM(respond)


@pytest.mark.parametrize(("answer", "kept", "rejected"), [
    ("same", [], [1, 2]),
    ("different", [1, 2], []),
    ("removed", [1], [2]),
    ("added", [2], [1]),
])
def test_replaced_pair_verdicts(tmp_path, answer, kept, rejected):
    write_pair(tmp_path, "red", "blue")
    vlm = in_place_vlm({("red", "blue"): answer, ("blue", "blue"): "same"})
    cfg = load_config(overrides=["semantics.enable_polish=false"])
    doc = run_semantics(tmp_path, cfg, vlm)
    assert sorted(c["id"] for c in doc["changes"]) == kept
    assert sorted(c["id"] for c in doc["rejected"]) == sorted([*rejected, 3])  # 3: "same"
    tags = [line["tag"] for line in map(json.loads, (tmp_path / "semantics" / "log.jsonl")
                                        .read_text().splitlines())]
    assert tags.count("judge_replaced:1+2") == 1 and "judge_in_place:3" in tags
    for c in doc["changes"] + doc["rejected"]:
        if c["id"] in (1, 2):
            assert c["evidence"]["vlm_replace_answer"] == answer
    for c in doc["rejected"]:  # "same", or the other type: the judge saw no such change
        assert c["evidence"]["reason_class"] == "judge_same"
    meta = json.loads((tmp_path / "semantics" / "meta.json").read_text())
    assert meta["replacements"] == [{"ids": [1, 2], "labels": ["wallet", "phone"],
                                     "answer": answer, "reason": "scripted"}]
    if answer == "different":  # the descriptions say what replaced what
        removed = next(c for c in doc["changes"] if c["id"] == 1)
        assert "replaced by a phone" in removed["description"]


def test_surface_named_appearance_change_is_rejected(tmp_path):
    """The bare bed sheet where the phone was later put (in-place answer "different"):
    named as a surface, so only the phone is kept."""
    write_pair(tmp_path, "grey", "blue", label_before="bed sheet")
    vlm = in_place_vlm({("grey", "blue"): "different", ("blue", "blue"): "same"})
    doc = run_semantics(tmp_path, load_config(overrides=["semantics.enable_polish=false"]), vlm)
    assert [c["id"] for c in doc["changes"]] == [2]
    sheet = next(c for c in doc["rejected"] if c["id"] == 1)
    assert "bare surface" in sheet["evidence"]["vlm_reason"]
    assert sheet["evidence"]["reason_class"] == "room_or_surface"
