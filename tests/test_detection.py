"""Tests for the detection channel (CPU, fake models, synthetic scenes).

Scene (OpenCV world, y down): a grey wall at z = 2 m filling every view, three before
cameras at z = 0 and three after cameras at z = 0.1 m, all looking along +z. A red box
stands in front of the wall in both recordings, a green box only before, a blue box only
after; a white poster on the wall (before) is replaced by an orange calendar at the same
place (after). The fake SAM 3 returns the rendered region of every object its phrase
names; the fake featurizer's tokens are the mean colour of each 16 px patch.
"""

import json

import numpy as np
import pytest
from scipy import ndimage

from roomdiff.changes import (
    detection_support,
    in_place_pairs,
    move_candidates,
    run_changes,
    source_support,
)
from roomdiff.config import load_config
from roomdiff.detection import (
    GATE_REASONS,
    VOCAB_PROMPT,
    BoxMask,
    Detection,
    DetObject,
    associate,
    build_vocabulary,
    cluster_detections,
    gate_reasons,
    keep_detection,
    lift_detection,
    load_detection_instances,
    main_parts,
    merge_frame_detections,
    normalise_phrase,
    parse_objects,
    phrase_key,
    run_detection,
    vocab_frames,
    vote_cosine,
)
from roomdiff.geometry import floor_obb
from roomdiff.instances import (
    Instance,
    View,
    clean_points,
    lift_mask,
    load_instances,
    run_instances,
)
from roomdiff.models import FakeFeaturizer, FakeSegmenter, FakeTextSegmenter
from roomdiff.recon import write_recon
from roomdiff.visibility import run_visibility
from roomdiff.vlm import FakeVLM

H, W = 240, 320
K = np.array([[160.0, 0.0, 159.5], [0.0, 160.0, 119.5], [0.0, 0.0, 1.0]], np.float32)
WALL_Z = 2.0
WALL = (90, 90, 90)
# name: (lo, hi, colour, recordings that hold it)
BOXES = {
    "red box": ((-0.55, -0.15, 1.4), (-0.25, 0.15, 1.7), (220, 30, 30), ("before", "after")),
    "green box": ((0.25, -0.15, 1.4), (0.55, 0.15, 1.7), (30, 200, 30), ("before",)),
    "blue box": ((-0.15, 0.25, 1.4), (0.15, 0.55, 1.7), (30, 30, 220), ("after",)),
}
# flat things painted on the wall: name -> (x range, y range, colour, recording)
PAINTED = {"poster": ((-0.2, 0.2), (-0.6, -0.35), (240, 240, 240), "before"),
           "calendar": ((-0.2, 0.2), (-0.6, -0.35), (240, 150, 30), "after")}
NAMES = [*BOXES, *PAINTED]  # region id = index + 1
CAMS = {"before": [(-0.3, 0.0, 0.0), (0.0, 0.05, 0.0), (0.3, 0.0, 0.0)],
        "after": [(-0.25, 0.0, 0.1), (0.05, -0.05, 0.1), (0.35, 0.0, 0.1)]}
VOCAB_ANSWER = json.dumps({"objects": ["Red box", "a green box", "the blue box", "poster",
                                       "calendar", "boxes", "floor", "wall", "object"]})
# opening_size 1: the rendered box faces are one voxel thick (real surfaces are 2-3)
SETTINGS = ["visibility.device=cpu", "visibility.voxel_size=0.02", "visibility.opening_size=1",
            "instances.device=cpu", "detection.enabled=true", "appearance.enabled=false"]
PARAMS = dict(load_config(overrides=SETTINGS).detection)


def pose(c) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = c
    return T


def render(c, boxes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z-depth, world hit points (H, W, 3) and box index + 1 per pixel (0 = wall) seen from
    camera centre c looking along +z."""
    c = np.asarray(c, np.float64)
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1)
    depth = np.full((H, W), WALL_Z - c[2])
    ids = np.zeros((H, W), np.int32)
    for k, (lo, hi) in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t0, t1 = (np.asarray(lo) - c) / d, (np.asarray(hi) - c) / d
        near, far = np.minimum(t0, t1).max(-1), np.maximum(t0, t1).min(-1)
        hit = (near < far) & (near > 0) & (near < depth)
        depth, ids = np.where(hit, near, depth), np.where(hit, k, ids)
    return depth.astype(np.float32), c + d * depth[..., None], ids


def make_recon() -> tuple[dict, dict[str, np.ndarray]]:
    """Recon dict of the scene and region id maps per recon frame (str index)."""
    data = {k: [] for k in ("rec", "files", "images", "cam2world", "K", "depth", "conf",
                            "valid")}
    labels = {}
    for rec_id, name in enumerate(("before", "after")):
        boxes = [(NAMES.index(n) + 1, (lo, hi)) for n, (lo, hi, _, recs) in BOXES.items()
                 if name in recs]
        for j, c in enumerate(CAMS[name]):
            depth, hit, ids = render(c, boxes)
            image = np.empty((H, W, 3), np.float64)
            image[:] = WALL
            for n, (_, _, colour, _) in BOXES.items():
                image[ids == NAMES.index(n) + 1] = colour
            for n, ((x0, x1), (y0, y1), colour, rec) in PAINTED.items():
                if rec != name:
                    continue
                inside = (ids == 0) & (hit[..., 0] >= x0) & (hit[..., 0] <= x1) & \
                    (hit[..., 1] >= y0) & (hit[..., 1] <= y1)
                image[inside] = colour
                ids = np.where(inside, NAMES.index(n) + 1, ids)
            labels[str(len(data["rec"]))] = ids
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


def write_scene(out) -> dict[str, np.ndarray]:
    recon, labels = make_recon()
    (out / "recon").mkdir(parents=True)
    write_recon(out / "recon" / "recon.npz", recon)
    (out / "recon" / "meta.json").write_text(json.dumps({"diagnostics": {}}))
    return labels


def fake_sam3(labels: dict[str, np.ndarray]) -> FakeTextSegmenter:
    """Finds the region of every object a phrase names ("boxes": every box, lower score)."""
    def respond(key, prompt, size):
        frame = key.split("[")[1].rstrip("]")  # "recon/recon.npz:images[5]"
        ids = labels[frame]
        names = list(BOXES) if prompt == "boxes" else [prompt]
        score = 0.8 if prompt == "boxes" else 0.9
        return [(ids == NAMES.index(n) + 1, score) for n in names
                if n in NAMES and (ids == NAMES.index(n) + 1).any()]
    return FakeTextSegmenter(respond)


# ------------------------------------------------------------------ vocabulary


def test_phrases_are_normalised_and_filtered():
    assert normalise_phrase("The Office Chairs") == "office chairs"
    assert normalise_phrase("a cup on the table") == "cup"
    assert normalise_phrase("wall") is None and normalise_phrase("door frame") is None
    assert normalise_phrase("Object") is None and normalise_phrase("furniture") is None
    assert normalise_phrase("person") is None and normalise_phrase("123") is None
    assert normalise_phrase("wall clock") == "wall clock"  # named after the object
    assert normalise_phrase("road") is None and normalise_phrase("blue sky") is None
    assert normalise_phrase("street lamp") == "street lamp"
    assert phrase_key("water bottles") == phrase_key("water bottle") == "water bottle"
    assert phrase_key("glass") == "glass"


def test_vocabulary_ranks_by_frames_dedups_and_caps():
    answers = [["Chair", "a table", "Cups", "floor"], ["chairs", "cup", "lamp"],
               ["chair", "lamp", "the window", "item"]]
    prompts, info = build_vocabulary(answers, 2, extra=["backpack", "chair", "wall"])
    assert prompts == ["chair", "cups", "backpack"]  # chair 3 frames; cups before lamp (order)
    assert info["counts"] == {"chair": 3, "cups": 2, "lamp": 2, "table": 1}
    assert info["capped"] == ["lamp", "table"] and info["extra"] == ["backpack"]
    assert set(info["dropped"]) == {"floor", "the window", "item"}
    assert build_vocabulary([], 40) == ([], {"counts": {}, "capped": [], "dropped": [],
                                             "extra": []})


def test_vocabulary_prompt_names_any_place():
    """The SceneDiff street pairs got no objects from a prompt that called every photo a
    room; the prompt now names other places too."""
    assert "a room, a shop, a street" in VOCAB_PROMPT
    assert "photo of a room" not in VOCAB_PROMPT


def test_vocabulary_answer_parsing():
    assert parse_objects('{"objects": ["sofa", "lamp"]}') == ["sofa", "lamp"]
    # cut off by the token limit while repeating itself: the listed strings still count
    assert parse_objects('{"objects": ["sofa", "coffee table", "remote", "rem') == \
        ["sofa", "coffee table", "remote"]
    assert parse_objects("I see a sofa.") == []


def test_vocabulary_frames_are_spread_and_sharp():
    frames = list(range(10, 22))
    sharp = [1.0, 5.0, 2.0, 1.0, 1.0, 9.0, 3.0, 1.0, 1.0, 1.0, 4.0, 2.0]
    assert vocab_frames(frames, 4, sharp) == [11, 15, 16, 20]
    assert vocab_frames(frames, 4, [None] * 12) == [11, 14, 17, 20]
    assert vocab_frames(frames[:2], 4, sharp[:2]) == [10, 11]
    assert vocab_frames([], 4, []) == []


# ------------------------------------------------------------------ detections


def square(x0, y0, x1, y1, shape=(60, 80)) -> np.ndarray:
    mask = np.zeros(shape, bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_detections_of_two_phrases_on_one_object_are_merged():
    chair = square(10, 10, 40, 40)
    nearly = square(11, 10, 40, 41)  # IoU ~0.93 with chair
    half = square(10, 10, 40, 25)  # IoU 0.5 with chair: another object (the seat)
    found = [("chair", chair, 0.9), ("office chair", nearly, 0.7), ("seat", half, 0.6),
             ("chair", nearly, 0.5), ("lamp", square(50, 5, 70, 50), 0.8)]
    dets = merge_frame_detections(found, 0.7, rec=1, frame=7)
    assert len(dets) == 3
    first = dets[0]
    assert first.votes == {"chair": 0.9, "office chair": 0.7} and first.score == 0.9
    assert np.array_equal(first.mask, chair) and (first.rec, first.frame) == (1, 7)
    assert first.region.crop.shape == (30, 30)  # only the box is kept
    assert [d.votes for d in dets[1:]] == [{"lamp": 0.8}, {"seat": 0.6}]
    assert keep_detection(chair, 0.6, 0.5, 100, 0.5)
    assert not keep_detection(chair, 0.4, 0.5, 100, 0.5)  # low score
    assert not keep_detection(square(0, 0, 5, 5), 0.9, 0.5, 100, 0.5)  # tiny
    assert not keep_detection(square(0, 0, 80, 40), 0.9, 0.5, 100, 0.5)  # 2/3 of the image


def full_iou(a, b) -> float:
    return float(np.count_nonzero(a & b) / np.count_nonzero(a | b))


def test_box_masks_match_the_full_masks():
    a, b = square(10, 10, 40, 40), square(30, 20, 60, 50)
    a[12, 12] = False  # holes stay holes
    ma, mb = BoxMask.of(a), BoxMask.of(b)
    assert ma.box == (10, 10, 39, 39) and ma.shape == (60, 80) and ma.area == a.sum()
    assert np.array_equal(ma.full(), a) and np.array_equal(mb.full(), b)
    assert ma.iou(mb) == pytest.approx(full_iou(a, b))
    union = ma.union(mb)
    assert union.box == (10, 10, 59, 49) and np.array_equal(union.full(), a | b)
    assert ma.iou(BoxMask.of(square(70, 50, 75, 55))) == 0.0  # boxes apart
    assert np.array_equal(ma.window((5, 5, 45, 45)), a[5:46, 5:46])


def test_lifting_inside_the_box_equals_lifting_the_full_mask():
    """The lifting works on the mask's box only (memory): same points in the same order
    as eroding and lifting the full-keyframe mask."""
    recon, labels = make_recon()
    inst = load_config().instances
    for frame, name in ((0, "green box"), (4, "blue box"), (1, "poster")):
        mask = labels[str(frame)] == NAMES.index(name) + 1
        det = Detection(0, frame, BoxMask.of(mask), 0.9, {name: 0.9})
        pts, cols = lift_detection(recon, det, 2, inst.lift_voxel, inst.lift_cell_m,
                                   inst.lift_link_m)
        eroded = ndimage.binary_erosion(mask, iterations=2)
        ref_pts, ref_cols = clean_points(*lift_mask(recon, frame, eroded), inst.lift_voxel)
        keep = main_parts(ref_pts, inst.lift_cell_m, inst.lift_link_m)
        assert len(pts) > 50
        assert np.array_equal(pts, ref_pts[keep].astype(np.float32))
        assert np.array_equal(cols, ref_cols[keep])


def plane(x0, x1, y0, y1, z, step=0.01) -> np.ndarray:
    xs, ys = np.arange(x0, x1, step), np.arange(y0, y1, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], 1).astype(np.float32)


def test_main_parts_drop_a_far_background_fragment():
    obj = plane(0, 0.2, 0, 0.2, 1.0)  # 400 points
    far = plane(0, 0.05, 0, 0.05, 3.0)  # 25 points of the wall behind it
    near = plane(0, 0.15, 0, 0.2, 1.06)  # a second face 6 cm behind: another large part
    keep = main_parts(np.vstack([obj, far]), 0.015, 0.03)
    assert keep[:400].all() and not keep[400:].any()
    assert main_parts(np.vstack([obj, near]), 0.015, 0.03).all()  # >= half the largest
    assert len(main_parts(np.zeros((0, 3)), 0.015, 0.03)) == 0


def test_clustering_keeps_a_book_on_a_table_apart():
    table = plane(0, 1.2, 0, 0.6, 0.0)
    view_a = plane(0, 0.7, 0, 0.6, 0.0)  # two partial views of the table
    view_b = plane(0.5, 1.2, 0, 0.6, 0.0)
    book = plane(0.3, 0.5, 0.2, 0.35, 0.02)  # 2 cm above the table top
    lamp = plane(3.0, 3.2, 0, 0.3, 0.0)
    groups = cluster_detections(
        [table, view_a, view_b, book, lamp],
        [{"table": 0.9}, {"table": 0.8}, {"desk": 0.7}, {"book": 0.9}, {"table": 0.9}],
        0.03, 0.3, 0.5)
    assert sorted(map(sorted, groups)) == [[0, 1, 2], [3], [4]]
    # the same containment with a shared phrase is one object (a partial view of it)
    assert len(cluster_detections([table, book], [{"table": 1}, {"table": 1}],
                                  0.03, 0.3, 0.5)) == 1


# ------------------------------------------------------------------ association


def det_object(rec, oid, points, votes, colour) -> DetObject:
    feat = np.asarray(colour, np.float32)
    obj = DetObject(rec, [], points.astype(np.float32), np.zeros((len(points), 3), np.uint8),
                    [], dict(votes), id=oid, obb=floor_obb(points))
    obj.feature = feat / np.linalg.norm(feat)
    return obj


def cube(centre, size=0.3, step=0.02) -> np.ndarray:
    r = np.arange(-size / 2, size / 2 + 1e-9, step)
    g = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    return g + np.asarray(centre)


def test_association_rules():
    red, blue = (1, 0, 0), (0, 0, 1)
    before = [det_object(0, 1, cube((0, 0, 0)), {"chair": 2.0}, red),
              det_object(0, 2, cube((1, 0, 0)), {"bottle": 1.0}, red),
              det_object(0, 3, cube((2, 0, 0)), {"box": 1.0}, red),
              det_object(0, 4, cube((3, 0, 0)), {"vase": 1.0}, red),
              det_object(0, 5, cube((5, 0, 0)), {"bag": 1.0}, red)]
    after = [det_object(1, 6, cube((0.02, 0, 0)), {"chair": 1.0, "stool": 1.5}, red),
             det_object(1, 7, cube((1.03, 0, 0)), {"bottle": 1.0}, blue),
             det_object(1, 8, cube((2, 0, 0.01)), {"plant": 1.0}, red),
             det_object(1, 9, cube((3, 0, 0)), {"vase": 1.0}, red),
             det_object(1, 10, cube((3.02, 0, 0)), {"vase": 0.5}, red),
             det_object(1, 11, cube((6.5, 0, 0)), {"bag": 1.0}, red),
             det_object(1, 12, cube((8, 0, 0)), {"cup": 1.0}, red)]
    before.append(det_object(0, 13, cube((8.01, 0, 0)), {"cup": 1.0}, red))
    before[-1].weak = True  # seen in one keyframe only: still counts as the cup's counterpart
    after.append(det_object(1, 14, cube((7.5, 0, 0)), {"mug": 1.0}, blue))
    after[-1].weak = True
    pairs = associate(before, after, PARAMS)
    decided = {o.id: (o.decision, o.partner) for o in [*before, *after]}
    assert vote_cosine({"chair": 2.0}, {"chair": 1.0, "stool": 1.5}) == pytest.approx(0.5547,
                                                                                      abs=1e-3)
    assert decided[1] == ("unchanged", 6) and decided[6] == ("unchanged", 1)  # vote cosine
    assert decided[2] == ("replaced", 7) and decided[7] == ("replaced", 2)  # other features
    assert decided[3] == ("replaced", 8) and decided[8] == ("replaced", 3)  # other label
    assert decided[4][0] == "unchanged"  # two fragments of one vase: both match it
    assert decided[9] == ("unchanged", 4) and decided[10] == ("unchanged", 4)
    assert decided[5] == ("unmatched", None) and decided[11] == ("unmatched", None)  # moved
    assert before[4].nearest == {"id": 11, "distance": 1.5, "feat_cos": 1.0}
    assert decided[12] == ("unchanged", 13)  # weak counterpart
    assert decided[14][0] == "unmatched"  # a weak object is never replaced
    logged = {(p["before"], p["after"]): p for p in pairs}
    assert logged[(1, 6)]["unchanged"] and not logged[(2, 7)]["unchanged"]
    assert logged[(13, 12)]["weak"] == [True, False]
    assert (5, 11) not in logged and (1, 7) not in logged  # not at the same place


# ------------------------------------------------------------------ stage


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    out = tmp_path_factory.mktemp("detection")
    labels = write_scene(out)
    cfg = load_config(overrides=SETTINGS)
    vlm = FakeVLM([VOCAB_ANSWER])
    meta = run_detection(out, cfg, vlm, fake_sam3(labels), FakeFeaturizer())
    return out, labels, cfg, meta, vlm


def test_detection_stage_finds_the_changed_objects(scene):
    out, _, _, meta, vlm = scene
    folder = out / "detection"
    vocab = json.loads((folder / "vocabulary.json").read_text())
    assert vocab["prompts"] == ["red box", "green box", "blue box", "poster", "calendar", "boxes"]
    assert len(vocab["answers"]) == 6 and vlm.calls == 6  # 3 keyframes per recording, all
    assert meta["objects"] == {"before": 3, "after": 3}
    doc = json.loads((folder / "clusters.json").read_text())
    objs = {(c["recording"], c["top_label"]): c for c in doc["clusters"]}
    red = objs[("before", "red box")]
    assert red["decision"] == "unchanged" and red["kind"] is None
    assert red["label_votes"] == {"red box": 2.7, "boxes": 2.4}  # merged phrases, 3 views
    assert objs[("after", "red box")]["partner"] == red["id"]
    green, blue = objs[("before", "green box")], objs[("after", "blue box")]
    assert (green["decision"], green["kind"]) == ("unmatched", "disappeared")
    assert (blue["decision"], blue["kind"]) == ("unmatched", "appeared")
    assert green["place"]["observing_views"] == 3 and green["place"]["see_through"] > 0.9
    poster, calendar = objs[("before", "poster")], objs[("after", "calendar")]
    assert (poster["decision"], poster["partner"]) == ("replaced", calendar["id"])
    assert calendar["kind"] == "appeared" and poster["kind"] == "disappeared"
    assert poster["place"]["surface_consistent"] > 0.9  # flat: the wall is still there
    assert len(green["views"]) == 3 and green["mean_score"] == pytest.approx(0.9)
    assert max(green["extents"]) == pytest.approx(0.3, abs=0.05)  # seen faces (erosion)
    assert green["place"]["min_camera_distance"] == pytest.approx(1.4, abs=0.2)
    association = json.loads((folder / "association.json").read_text())
    assert {d["decision"] for d in association["decisions"]} == {"unchanged", "unmatched",
                                                                  "replaced"}
    assert meta["candidates"] == {"disappeared": {"before": 2, "after": 0},
                                  "appeared": {"before": 0, "after": 2}}
    assert meta["gate"]["dropped"] == {"before": 0, "after": 0} and association["gate"] == []
    assert green["gate"] == [] and green["label_count"] == 1


def test_gate_reasons():
    params = {"min_observing_views": 3, "min_mean_score": 0.65, "max_label_count": 2}
    assert gate_reasons(3, 0.65, 2, params) == []  # the limits themselves pass
    assert gate_reasons(0, 0.9, 1, params) == ["no_observing_views"]
    assert gate_reasons(2, 0.9, 1, params) == ["few_observing_views"]
    assert gate_reasons(8, 0.6, 1, params) == ["low_score"]
    assert gate_reasons(8, 0.9, 3, params) == ["common_label"]  # three look-alike bottles
    assert gate_reasons(1, 0.5, 9, params) == ["few_observing_views", "low_score",
                                                "common_label"]


def copy_run(scene, tmp_path):
    import shutil

    out = scene[0]
    copy = tmp_path / "run"
    shutil.copytree(out / "recon", copy / "recon")
    shutil.copytree(out / "detection", copy / "detection")
    return copy


@pytest.mark.parametrize(("overrides", "reasons"), [
    (["detection.min_observing_views=4", "detection.min_mean_score=0.95"],
     ["few_observing_views", "low_score"]),
    (["detection.max_label_count=0"], ["common_label"]),
])
def test_gate_drops_candidates_before_the_judge(scene, tmp_path, overrides, reasons):
    """Every candidate of the scene is seen in 3 views of the other recording with a SAM 3
    score of 0.9 and a label of its own; stricter limits drop all four, with the reasons in
    association.json, clusters.json and meta.json, and the loader adds no instance."""
    copy = copy_run(scene, tmp_path)
    cfg = load_config(overrides=[*SETTINGS, "semantics.model=fake", *overrides])
    meta = run_detection(copy, cfg, featurizer=FakeFeaturizer())
    assert meta["candidates"] == {"disappeared": {"before": 0, "after": 0},
                                  "appeared": {"before": 0, "after": 0}}
    assert meta["gate"]["dropped"] == {"before": 2, "after": 2}
    assert meta["gate"]["reasons"] == {k: 4 * (k in reasons) for k in GATE_REASONS}
    gate = json.loads((copy / "detection" / "association.json").read_text())["gate"]
    assert sorted(g["label"] for g in gate) == ["blue box", "calendar", "green box", "poster"]
    assert all(g["reasons"] == reasons for g in gate)
    green = next(g for g in gate if g["label"] == "green box")
    assert green["observing_views"] == 3 and green["label_count"] == 1
    assert green["mean_score"] == pytest.approx(0.9) and green["see_through"] > 0.9
    doc = json.loads((copy / "detection" / "clusters.json").read_text())
    for c in doc["clusters"]:
        assert c["kind"] is None
        assert c["gate"] == (reasons if c["decision"] != "unchanged" and not c["weak"] else None)
    extra, loaded = load_detection_instances(copy, [[], []], 0.03, 0.5)
    assert extra == [[], []] and loaded["candidates"] == 0


def test_vocabulary_cache_key_holds_the_prompt(scene, monkeypatch, tmp_path):
    """A vocabulary asked with another prompt is asked again (one VLM call per keyframe);
    the same answers give the same prompts, so the SAM 3 detections are reused."""
    from roomdiff import detection

    out = scene[0]
    stored = json.loads((out / "detection" / "vocabulary.json").read_text())
    assert stored["key"]["prompt"] == VOCAB_PROMPT
    copy = copy_run(scene, tmp_path)
    loaded = []

    def fake_load(params):
        loaded.append(FakeVLM([VOCAB_ANSWER]))
        return loaded[-1]

    monkeypatch.setattr("roomdiff.vlm.load_vlm", fake_load)
    monkeypatch.setattr(detection, "VOCAB_PROMPT", "This is a photo of a room. " + VOCAB_PROMPT)
    cfg = load_config(overrides=[*SETTINGS, "semantics.model=fake"])
    meta = run_detection(copy, cfg, featurizer=FakeFeaturizer())
    assert not meta["vocabulary"]["cached"] and loaded[0].calls == 6
    assert meta["sam3"]["cached"]


def test_cached_vocabulary_and_detections_are_reused(scene, monkeypatch, tmp_path):
    import shutil

    out, _, cfg, meta, _ = scene
    copy = tmp_path / "run"
    shutil.copytree(out / "recon", copy / "recon")
    shutil.copytree(out / "detection", copy / "detection")

    def no_model(*args, **kwargs):
        raise AssertionError("a model was loaded although the cache matches")

    monkeypatch.setattr("roomdiff.vlm.load_vlm", no_model)
    monkeypatch.setattr("roomdiff.models.Sam3TextSegmenter", no_model)
    cfg = load_config(overrides=[*SETTINGS, "semantics.model=fake"])
    again = run_detection(copy, cfg, featurizer=FakeFeaturizer())
    assert again["vocabulary"]["cached"] and again["sam3"]["cached"]
    assert again["objects"] == meta["objects"] and again["candidates"] == meta["candidates"]


def test_disabled_stage_writes_nothing(tmp_path):
    write_scene(tmp_path)
    cfg = load_config(overrides=[*SETTINGS, "detection.enabled=false"])
    assert run_detection(tmp_path, cfg, FakeVLM(), FakeTextSegmenter(), FakeFeaturizer()) == \
        {"enabled": False}
    assert not (tmp_path / "detection").exists()


def test_loader_skips_covered_and_unchanged_objects(scene):
    out = scene[0]
    doc = json.loads((out / "detection" / "clusters.json").read_text())
    green = next(c for c in doc["clusters"] if c["top_label"] == "green box")
    with np.load(out / "detection" / "clusters_points.npz") as f:
        pts = f["points"][f["cluster"] == green["id"]]
    cover = Instance(0, "disappeared", [], pts, pts + [0.005, 0, 0],
                     np.zeros((len(pts), 3), np.uint8), [], id=1)
    extra, meta = load_detection_instances(out, [[cover], []], 0.03, 0.5)
    assert [c["label"] for c in meta["covered"]] == ["green box"]
    assert meta["covered"][0]["instance"] == 1 and meta["candidates"] == 4
    (poster,) = extra[0]
    assert {i.evidence["detection"]["top_label"] for i in extra[1]} == {"blue box", "calendar"}
    assert poster.source == "detection" and poster.kind == "disappeared" and poster.blobs == []
    assert len(poster.views) == 3 and all(isinstance(v, View) for v in poster.views)
    assert poster.views[0].mask.shape == (H, W) and poster.views[0].score == pytest.approx(0.9)
    ev = poster.evidence
    assert ev["observing_views"] == 3 and ev["occ_points"] == len(poster.points)
    assert ev["detection"]["decision"] == "replaced" and ev["detection"]["n_views"] == 3
    assert set(ev["detection"]) == {"object", "top_label", "label_votes", "mean_score",
                                    "n_views", "observing_views", "decision", "partner",
                                    "surface_consistent", "see_through"}
    assert {c["kind"] for c in doc["clusters"]} == {None, "disappeared", "appeared"}


def test_instances_and_changes_with_detection_candidates(scene):
    out, labels, cfg, _, _ = scene
    run_visibility(out, cfg)
    meta = run_instances(out, cfg, FakeSegmenter(labels), FakeFeaturizer())
    # the boxes are geometric changes (M3 blobs): their detections are covered
    assert sorted(c["label"] for c in meta["detection"]["covered"]) == ["blue box", "green box"]
    assert [meta["sources"][name]["detection"] for name in ("before", "after")] == [1, 1]
    for name in ("before", "after"):
        data = load_instances(out, name)
        det = [r for r in data["instances"] if r["source"] == "detection"]
        assert len(det) == 1 and det[0]["blob_ids"] == [] and det[0]["n_frame_masks"] == 3
        assert det[0]["id"] == len(data["instances"])  # appended after the geometric ones
    doc = run_changes(out, cfg)
    det = [c for c in doc["changes"]
           if (c["before"] or c["after"])["source"] == "detection"]
    removed = next(c for c in det if c["type"] == "removed")
    added = next(c for c in det if c["type"] == "added")
    assert removed["replaced_by"] == added["id"] and added["replaces"] == removed["id"]
    assert removed["evidence"]["detection"]["top_label"] == "poster"
    geo = [c for c in doc["changes"] if c not in det]
    assert sorted(c["type"] for c in geo) == ["added", "removed"]  # the two boxes
    # same evidence, so the detection change ranks lower: x mean SAM 3 score (0.9)
    assert removed["confidence"] < max(c["confidence"] for c in geo)


def test_detection_confidence_factor():
    geo = {"evidence": {"free_rays": 3}}
    det = {"evidence": {"detection": {"mean_score": 0.72}}}
    app = {"evidence": {"appearance": {"mean_score": 1.0}}}
    params = dict(load_config().changes)
    assert detection_support(geo) == 1.0 and detection_support(det) == 0.72
    assert source_support(det, params) == 0.72
    assert source_support(app, params) == pytest.approx(0.75)
    assert detection_support({"evidence": {"detection": {"mean_score": 1.3}}}) == 1.0


def test_detection_ids_follow_both_recordings():
    det = Detection(0, 3, BoxMask.of(square(10, 10, 40, 40)), 0.9, {"chair": 0.9})
    assert det.box == (10, 10, 39, 39)


def record(kind, source, centre, size=0.1):
    return {"kind": kind, "source": source,
            "obb": {"center": list(centre), "extents": [size] * 3, "yaw_deg": 0.0}}


def test_in_place_pairs_stay_within_one_source():
    """living_room_35: a detection ('black pen') at the old place of a moved mug paired
    with the mug's appearance instance and split the move. Each source now pairs only
    with itself; geometry instances never pair in place."""
    before = [record("disappeared", "appearance", (0, 0, 0)),   # the mug before
              record("disappeared", "detection", (1, 0, 0)),
              record("disappeared", "geometry", (2, 0, 0)),
              record("unchanged", "detection", (3, 0, 0))]
    after = [record("appeared", "detection", (0.01, 0, 0)),     # a pen at the mug's place
             record("appeared", "detection", (1.01, 0, 0)),
             record("appeared", "appearance", (1.0, 0, 0.01)),
             record("appeared", "geometry", (2.01, 0, 0)),
             record("appeared", "appearance", (5, 0, 0))]       # the mug after
    pairs = in_place_pairs(before, after, 0.3)
    assert [(i, j) for i, j, _ in pairs] == [(1, 1)]


def test_detection_instances_never_enter_the_moved_matching():
    """All 11 moved changes with a detection side on the dev pairs were false (look-alike
    price tags, bottles): detection instances stay removed / added."""
    records = [record("disappeared", "geometry", (0, 0, 0)),
               record("disappeared", "detection", (1, 0, 0)),
               record("disappeared", "appearance", (2, 0, 0)),
               record("appeared", "appearance", (3, 0, 0)),
               {"kind": "disappeared"}]  # an old record without a source: geometry
    assert move_candidates(records, "disappeared", set()) == [0, 2, 4]
    assert move_candidates(records, "disappeared", {2}) == [0, 4]  # in an in-place pair
    assert move_candidates(records, "appeared", set()) == [3]
