"""Tests for the optional SAM 3 re-check (CPU only, FakeTextSegmenter, no weights).

Synthetic scene (OpenCV world, y down): a wall at z = 2 m, flat 20 cm squares 10 cm in
front of it, four cameras per recording at z = 0 (before) / 0.1 m (after) looking along
+z. Before: red box R and blue cup at C0; after: green ball G, blue cup at C1 and a chair
U seen only after (unverified). Changes: removed red box, added green ball, moved blue
cup, unverified chair. The fake SAM 3 "finds" an object in a keyframe when the test's
`world` says it is in that recording; its mask is the object's projection.
"""

import json

import numpy as np
import pytest

from roomdiff.config import load_config
from roomdiff.instances import encode_rle
from roomdiff.models import FakeTextSegmenter
from roomdiff.recheck import (
    decide,
    description_noun,
    project_region,
    prompt_for,
    run_recheck,
)
from roomdiff.recon import write_recon

H, W = 120, 160
K = np.array([[100.0, 0.0, 79.5], [0.0, 100.0, 59.5], [0.0, 0.0, 1.0]], np.float32)
WALL_Z, OBJ_Z = 2.0, 1.9
CAMS = {"before": [(x, 0.0) for x in (-0.3, -0.1, 0.1, 0.3)],
        "after": [(x, 0.1) for x in (-0.25, -0.05, 0.15, 0.35)]}
# (x0, y0) corner of each 20 cm square
SQUARES = {"red": (-0.5, -0.3), "green": (0.3, -0.3), "cup0": (-0.1, 0.1), "cup1": (0.25, 0.1),
           "chair": (-0.5, 0.1)}
PRESENT = {"before": ["red", "cup0"], "after": ["green", "cup1", "chair"]}
INSTANCES = {"before": {1: "red", 2: "cup0"}, "after": {1: "green", 2: "cup1", 3: "chair"}}
PARAMS = {"score_min": 0.5, "overlap_min": 0.3, "depth_tol": 0.15, "reject_views": 2,
          "contradicted_factor": 0.6, "unsupported_factor": 0.9}
ON = ["recheck.enabled=true", "recheck.device=cpu"]


def square_points(name, z=OBJ_Z):
    x0, y0 = SQUARES[name]
    xs, ys = np.meshgrid(np.arange(x0, x0 + 0.2001, 0.01), np.arange(y0, y0 + 0.2001, 0.01))
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], 1)


def square_box(name, cam, z=OBJ_Z):
    """Pixel box (u0, v0, u1, v1) of a square seen from camera position `cam` (x, z)."""
    x0, y0 = SQUARES[name]
    d = z - cam[1]
    return (K[0, 2] + K[0, 0] * (x0 - cam[0]) / d, K[1, 2] + K[1, 1] * y0 / d,
            K[0, 2] + K[0, 0] * (x0 + 0.2 - cam[0]) / d, K[1, 2] + K[1, 1] * (y0 + 0.2) / d)


def square_mask(name, cam, z=OBJ_Z):
    u0, v0, u1, v1 = square_box(name, cam, z)
    v, u = np.mgrid[0:H, 0:W]
    return (u >= np.ceil(u0)) & (u <= np.floor(u1)) & (v >= np.ceil(v0)) & (v <= np.floor(v1))


def side(iid, name):
    pts = square_points(name)
    lo, hi = pts.min(0), pts.max(0)
    return {"instance_id": iid, "centroid": pts.mean(0).round(4).tolist(),
            "aabb": [lo.tolist(), hi.tolist()], "extents": (hi - lo).tolist(),
            "obb": {"center": pts.mean(0).tolist(), "extents": (hi - lo).tolist(), "yaw_deg": 0.0},
            "n_points": len(pts), "blob_ids": [iid], "source": "geometry"}


def change(cid, kind, label, before=None, after=None, conf=0.8):
    ev = {"free_rays": 50, "observing_views": 4, "mask_score": 0.9,
          "vlm_verdict": "confirmed", "vlm_reason": "scripted"}
    if kind == "unverified":
        ev |= {"seen_in": "after", "vlm_verdict": "unclear"}
    return {"id": cid, "type": kind, "label": label, "confidence": conf,
            "before": before, "after": after, "transform": None, "evidence": ev,
            "attributes": "", "object_description": f"A {label}.", "motion_kind": None,
            "description": f"The {label} changed.", "crops": {}}


def make_scene(out):
    """recon.npz, instance points / per-keyframe masks and an M5-like changes.json."""
    data = {k: [] for k in ("rec", "files", "images", "cam2world", "K", "depth", "conf", "valid")}
    stored = {"before": [], "after": []}
    for rec_id, rec in enumerate(("before", "after")):
        for j, cam in enumerate(CAMS[rec]):
            T = np.eye(4)
            T[:3, 3] = [cam[0], 0.0, cam[1]]
            depth = np.full((H, W), WALL_Z - cam[1], np.float32)
            for name in PRESENT[rec]:
                depth[square_mask(name, cam)] = OBJ_Z - cam[1]
            file = f"{rec}_{j:05d}.jpg"
            for iid, name in INSTANCES[rec].items():
                stored[rec].append((file, iid, 0.95, encode_rle(square_mask(name, cam))))
            data["rec"].append(rec_id)
            data["files"].append(file)
            data["images"].append(np.full((H, W, 3), 128, np.uint8))
            data["cam2world"].append(T)
            data["K"].append(K)
            data["depth"].append(depth)
            data["conf"].append(np.full((H, W), 0.9))
            data["valid"].append(np.ones((H, W), bool))
    recon = {k: np.stack(v) for k, v in data.items()}
    recon["metric_scale"] = 1.0
    (out / "recon").mkdir(parents=True)
    write_recon(out / "recon" / "recon.npz", recon)
    folder = out / "instances"
    folder.mkdir()
    for rec in ("before", "after"):
        ids = list(INSTANCES[rec])
        pts = [square_points(INSTANCES[rec][i]) for i in ids]
        np.savez_compressed(folder / f"{rec}_points.npz", points=np.concatenate(pts),
                            colors=np.zeros((sum(map(len, pts)), 3), np.uint8),
                            instance=np.concatenate([np.full(len(p), i) for i, p in zip(ids, pts)]))
        rows = stored[rec]
        np.savez_compressed(folder / f"{rec}_masks.npz",
                            keyframe=np.array([r[0] for r in rows]),
                            instance_id=np.array([r[1] for r in rows], np.int32),
                            score=np.array([r[2] for r in rows], np.float32),
                            rle=np.array([r[3]["counts"] for r in rows]),
                            size=np.array([H, W]))
    moved = change(3, "moved", "blue cup", side(2, "cup0"), side(2, "cup1"), conf=0.7)
    moved["transform"] = {"translation_m": 0.35, "rotation_deg": None}
    doc = {"scene": {"before": "v1.mp4", "after": "v2.mp4", "units": "metres (approx.)"},
           "changes": [moved, change(1, "removed", "red box", before=side(1, "red")),
                       change(2, "added", "green ball", after=side(1, "green"), conf=0.9),
                       change(4, "unverified", "chair", after=side(3, "chair"), conf=0.4)],
           "rejected": [change(9, "removed", "wall", before=side(2, "cup0"), conf=0.5)],
           "summary": "3 changes: 1 moved, 1 added, 1 removed; 1 unverified. Things moved."}
    doc["rejected"][0]["evidence"]["vlm_verdict"] = "rejected"
    (out / "changes.json").write_text(json.dumps(doc, indent=2))
    return doc


LABELS = {"red box": ["red"], "green ball": ["green"], "blue cup": ["cup0", "cup1"]}


def fake_sam(world, score=0.9, frames=None):
    """FakeTextSegmenter that finds the objects `world[recording]` (square names) named by
    the prompt in every keyframe (or only in the recon `frames` given)."""
    def respond(key, prompt, size):
        frame = int(key.split("[")[1].rstrip("]"))  # "recon/recon.npz:images[5]"
        rec = "before" if frame < 4 else "after"
        if frames is not None and frame not in frames:
            return []
        cam = CAMS[rec][frame % 4]
        return [(square_mask(name, cam), score) for name in LABELS.get(prompt, [])
                if name in world[rec]]
    return FakeTextSegmenter(respond)


@pytest.fixture
def scene(tmp_path):
    make_scene(tmp_path)
    return tmp_path


def by_id(doc):
    return {c["id"]: c for c in [*doc["changes"], *doc["rejected"]]}


# ------------------------------------------------------------------ decision policy


def view(found, place=None, rec="after"):
    return {"recording": rec, "found": found, **({"place": place} if place else {})}


def test_decide_policy():
    own_ok, own_miss = [view(True, rec="before")], [view(False, rec="before")]
    absent = [view(False, "old")] * 2
    assert decide("removed", own_ok, absent, PARAMS) == ("supported", 1.0, False)
    assert decide("removed", own_miss, absent, PARAMS) == ("unsupported", 0.9, False)
    assert decide("removed", own_ok, [view(True, "old"), view(False, "old")], PARAMS) == \
        ("contradicted", 0.6, False)  # one view only lowers the confidence
    assert decide("removed", own_ok, [view(True, "old")] * 2, PARAMS) == \
        ("contradicted", 0.6, True)
    assert decide("removed", own_miss, [view(True, "old")] * 2, PARAMS) == \
        ("contradicted", 0.54, True)
    assert decide("added", [], [view(True, "new")] * 3, {**PARAMS, "reject_views": 4}) == \
        ("contradicted", 0.6, False)
    # moved: one place alone may be a sibling (the other curtain), both places contradict
    assert decide("moved", own_ok, [view(True, "old")] * 2, PARAMS) == ("supported", 1.0, False)
    assert decide("moved", own_ok, [view(True, "old"), view(True, "new", "before")], PARAMS) == \
        ("contradicted", 0.6, True)
    assert decide("removed", [], [], PARAMS) == ("skipped", 1.0, False)
    # without own views the place alone decides
    assert decide("removed", [], absent, PARAMS) == ("supported", 1.0, False)


def test_prompt_falls_back_to_the_description_noun():
    assert prompt_for({"label": "office chair"}) == "office chair"
    assert prompt_for({"label": "grey fabric", "object_description":
                       "A crumpled black jacket lying on the floor."}) == "jacket"
    assert prompt_for({"label": "object", "object_description": "A small, black wallet."}) \
        == "wallet"
    assert prompt_for({"label": "leather", "object_description": "Some brown leather."}) is None
    assert prompt_for({"label": "object", "object_description": ""}) is None
    assert description_noun("The cardboard box with colourful labels.") == "box"


def test_place_region_follows_the_points_without_growing(scene):
    from roomdiff.recon import load_recon

    recon = load_recon(scene / "recon" / "recon.npz")
    truth = square_mask("red", CAMS["after"][1])  # recon frame 5
    region, zsurf = project_region(square_points("red"), 5, recon, "place", 0.01)
    assert (region & truth).sum() / (region | truth).sum() > 0.8
    assert np.allclose(zsurf[region], OBJ_Z - 0.1, atol=0.01)
    assert np.isinf(zsurf[~region]).all()
    # the object is not in the after recording: no depth-consistent (surface) points there
    assert not project_region(square_points("red"), 5, recon, "surface", 0.01)[0].any()
    assert project_region(square_points("red"), 1, recon, "surface", 0.01)[0].any()


# ------------------------------------------------------------------ stage


def test_consistent_scene_is_supported(scene):
    fake = fake_sam(PRESENT)
    doc = run_recheck(scene, load_config(overrides=ON), fake)
    changes = by_id(doc)
    for cid in (1, 2, 3):
        rc = changes[cid]["evidence"]["recheck"]
        assert rc["verdict"] == "supported", rc["reason"]
        assert rc["factor"] == 1.0 and rc["own"]["found"] and rc["other"]["found"] is False
        assert all(v["overlap"] > 0.9 for v in rc["own"]["views"])
    assert [c["confidence"] for c in doc["changes"] if c["id"] in (1, 2, 3)] == [0.7, 0.8, 0.9]
    moved = changes[3]["evidence"]["recheck"]
    assert {v["recording"] for v in moved["own"]["views"]} == {"before", "after"}
    assert {v["place"] for v in moved["other"]["views"]} == {"old", "new"}
    assert len(moved["other"]["views"]) == 4 and len(changes[1]["evidence"]["recheck"]
                                                     ["other"]["views"]) == 2
    # unverified changes are skipped, the M5 rejected list is left alone
    assert changes[4]["evidence"]["recheck"]["verdict"] == "skipped"
    assert changes[4]["confidence"] == 0.4
    assert "recheck" not in changes[9]["evidence"]
    assert [c["id"] for c in doc["rejected"]] == [9]
    assert doc["summary"].startswith("3 changes: 1 moved, 1 added, 1 removed; 1 unverified.")
    meta = json.loads((scene / "recheck" / "meta.json").read_text())
    assert meta["counts"] == {"supported": 3, "unsupported": 0, "contradicted": 0,
                              "skipped": 1, "rejected": 0}
    assert meta["images"] == fake.images <= 8
    log = (scene / "recheck" / "log.jsonl").read_text().splitlines()
    assert len(log) == meta["queries"] == len(fake.queries)
    report = (scene / "report.md").read_text()
    assert "SAM 3 re-check supported: 'red box' found on the object in 2 of 2 own views " \
        "(0.90), absent from the old place in 2 after views" in report


def test_unsupported_label_is_evidence_only_by_default(scene):
    world = {"before": ["red", "cup0"], "after": ["cup1"]}  # SAM 3 misses the green ball
    doc = run_recheck(scene, load_config(overrides=ON), fake_sam(world))
    added = by_id(doc)[2]
    assert added["evidence"]["recheck"]["verdict"] == "unsupported"
    assert added["evidence"]["recheck"]["own"]["found"] is False
    assert added["confidence"] == pytest.approx(0.9)  # unsupported_factor 1.0 by default
    assert added in doc["changes"]


def test_unsupported_factor_lowers_the_confidence_when_set(scene):
    world = {"before": ["red", "cup0"], "after": ["cup1"]}
    doc = run_recheck(scene, load_config(overrides=ON + ["recheck.unsupported_factor=0.9"]),
                      fake_sam(world))
    added = by_id(doc)[2]
    assert added["evidence"]["recheck"]["verdict"] == "unsupported"
    assert added["confidence"] == pytest.approx(0.9 * 0.9)


def test_one_contradicting_view_lowers_the_confidence(scene):
    world = {"before": ["red", "cup0"], "after": ["red", "green", "cup1"]}
    first = run_recheck(scene, load_config(overrides=ON + ["recheck.max_views=1"]),
                        fake_sam(world))
    removed = by_id(first)[1]
    assert removed["evidence"]["recheck"]["verdict"] == "contradicted"
    assert removed["evidence"]["recheck"]["other"]["found_views"] == 1
    assert removed["confidence"] == pytest.approx(0.8 * 0.6)
    assert removed in first["changes"] and "recheck_verdict" not in removed["evidence"]


def test_strong_contradiction_rejects_and_reruns_start_from_the_backup(scene):
    m5 = (scene / "changes.json").read_text()
    world = {"before": ["red", "cup0"], "after": ["red", "green", "cup1"]}
    doc = run_recheck(scene, load_config(overrides=ON), fake_sam(world))
    removed = by_id(doc)[1]
    assert removed in doc["rejected"] and removed not in doc["changes"]
    ev = removed["evidence"]
    assert ev["recheck_verdict"] == "contradicted" and ev["vlm_verdict"] == "confirmed"
    assert ev["recheck"]["other"]["found_views"] == 2
    assert all(v["at_place"] == 1.0 for v in ev["recheck"]["other"]["views"])
    assert removed["confidence"] == pytest.approx(0.48)
    assert doc["summary"].startswith("2 changes: 1 moved, 1 added, 0 removed; 1 unverified. "
                                     "Things moved. A SAM 3 re-check later rejected change 1")
    report = (scene / "report.md").read_text()
    assert "- **1** removed *red box*, confidence 0.48: SAM 3 re-check: 'red box'" in report
    assert (scene / "recheck" / "changes_m5.json").read_text() == m5
    # a second run starts from the M5 backup: same result, factors not applied twice
    again = run_recheck(scene, load_config(overrides=ON), fake_sam(world))
    assert again == doc


def test_detection_behind_the_place_is_no_contradiction(scene):
    """The red box hung 40 cm in front of the wall; SAM 3 finds 'a red box' on the wall
    behind its place in the after views (a poster): the depth test keeps it apart."""
    from roomdiff.recon import load_recon, write_recon

    recon = load_recon(scene / "recon" / "recon.npz")
    for f, cam in enumerate(CAMS["before"]):  # the box is at z = 1.6 in the before views
        recon["depth"][f][square_mask("red", cam, 1.6)] = 1.6 - cam[1]
    write_recon(scene / "recon" / "recon.npz", recon)
    pts = np.load(scene / "instances" / "before_points.npz")
    points = pts["points"].copy()
    points[pts["instance"] == 1, 2] = 1.6
    np.savez_compressed(scene / "instances" / "before_points.npz", points=points,
                        colors=pts["colors"], instance=pts["instance"])

    def respond(key, prompt, size):
        frame = int(key.split("[")[1].rstrip("]"))
        if prompt != "red box":
            return []
        rec = "before" if frame < 4 else "after"  # after: a poster on the wall behind the
        return [(square_mask("red", CAMS[rec][frame % 4], 1.6), 0.9)]  # place, same outline

    doc = run_recheck(scene, load_config(overrides=ON), FakeTextSegmenter(respond))
    rc = by_id(doc)[1]["evidence"]["recheck"]
    assert rc["verdict"] == "supported"
    views = rc["other"]["views"]
    assert len(views) == 2 and all(v["overlap"] > 0.8 and v["at_place"] == 0.0 for v in views)


def test_disabled_stage_leaves_changes_json_byte_identical(scene):
    from typer.testing import CliRunner

    from roomdiff.cli import app

    before = (scene / "changes.json").read_bytes()
    fake = fake_sam(PRESENT)
    assert run_recheck(scene, load_config(), fake) is None  # default: enabled false
    result = CliRunner().invoke(app, ["recheck", "--out", str(scene)])
    assert result.exit_code == 0, result.output
    assert "disabled" in result.output
    assert (scene / "changes.json").read_bytes() == before
    assert fake.images == 0 and not (scene / "recheck").exists()
    assert not (scene / "report.md").exists()


def test_recheck_needs_the_m5_document(scene):
    doc = json.loads((scene / "changes.json").read_text())
    del doc["rejected"]
    (scene / "changes.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="roomdiff describe"):
        run_recheck(scene, load_config(overrides=ON), fake_sam(PRESENT))
