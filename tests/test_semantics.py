"""Tests for M5 semantics with a scripted FakeVLM (CPU only, no weights).

The fake answers from the colours of the crops: red = a bag, green = a hairbrush,
grey = the wall (background). Same-object questions are confirmed when both crops have
the same non-grey colour.
"""

import json

import numpy as np
import pytest
from PIL import Image

from roomdiff.config import load_config
from roomdiff.geometry import floor_obb, obb_gap, rotation_y
from roomdiff.semantics import (
    SceneFrame,
    head_noun,
    labels_match,
    normalise_label,
    numbers_preserved,
    polish,
    run_semantics,
    template_description,
)
from roomdiff.vlm import JSON_RETRY, FakeVLM, extract_json

COLOURS = {"red": (210, 30, 30), "green": (30, 200, 40), "grey": (128, 128, 128)}
NAMES = {"red": "a Red Bag", "green": "the hairbrush", "grey": "wall"}


def colour(image: Image.Image) -> str:
    r, g, _ = np.asarray(image, float).reshape(-1, 3).mean(0)
    return "red" if r > 150 and g < 100 else "green" if g > 150 and r < 100 else "grey"


def responder(images, prompt):
    """Scripted VLM answers keyed on the prompt type and the crop colours."""
    cols = [colour(im) for im in images]
    if '"label"' in prompt:
        answer = {"label": NAMES[cols[0]], "attributes": cols[0], "description": "A thing."}
        return f"Sure! ```json\n{json.dumps(answer)}\n```"
    if "same physical object" in prompt:
        same = len(set(cols)) == 1 and cols[0] != "grey"
        return json.dumps({"verdict": "confirmed" if same else "rejected", "reason": "colours"})
    if '"motion"' in prompt:
        return '{"motion": "rigid", "reason": "kept its shape"}'
    if "distinct movable object" in prompt:
        ok = cols[0] != "grey"
        return json.dumps({"verdict": "confirmed" if ok else "rejected",
                           "reason": "an object" if ok else "only a plain wall"})
    if "added between" in prompt:
        return '{"verdict": "unclear", "reason": "the place is dark"}'
    if "absent from that place in image 2" in prompt:
        return '{"verdict": "confirmed", "reason": "gone"}'
    if '"sentences"' in prompt:
        return "I cannot do that."
    if '"summary"' in prompt:
        return '{"summary": "A red bag moved and a hairbrush may have appeared."}'
    return "?"


def crop(root, name, col):
    (root / "crops").mkdir(exist_ok=True)
    Image.new("RGB", (1000, 600), COLOURS[col]).save(root / "crops" / f"{name}.jpg")
    return f"crops/{name}.jpg"


def side(iid, centroid, half=0.1, n=500):
    c = np.asarray(centroid, float)
    return {"instance_id": iid, "centroid": c.tolist(), "aabb": [(c - half).tolist(),
            (c + half).tolist()], "extents": [2 * half] * 3,
            "obb": {"center": c.tolist(), "extents": [2 * half] * 3, "yaw_deg": 0.0},
            "n_points": n, "blob_ids": [iid]}


def change(cid, kind, before=None, after=None, crops=None, conf=0.8, views=5, seen_in=None):
    ev = {"free_rays": 100, "free_frames_mean": 3.0, "observing_views": views,
          "occ_points": 900, "mask_score": 0.9, "view_consistency": 0.8, "vlm_verdict": None}
    if seen_in:
        ev["seen_in"] = seen_in
    return {"id": cid, "type": kind, "label": "object", "confidence": conf, "before": before,
            "after": after, "transform": None, "evidence": ev, "description": "M4 text.",
            "crops": crops or {}}


def write_doc(root, changes):
    doc = {"scene": {"before": "a.mp4", "after": "b.mp4", "units": "metres (approx.)"},
           "changes": changes, "summary": "M4 summary."}
    (root / "changes.json").write_text(json.dumps(doc))


@pytest.fixture
def split_scene(tmp_path):
    """removed bag (1) + after-only bag 1 m away (2), background (3), unclear added
    hairbrush (4), removed hairbrush far from it (5)."""
    write_doc(tmp_path, [
        change(1, "removed", before=side(1, [0, 0, 2]),
               crops={"before": crop(tmp_path, "1_before", "red"),
                      "after": crop(tmp_path, "1_after", "grey")}, conf=0.9),
        change(2, "unverified", after=side(1, [1, 0, 2]), seen_in="after", conf=0.4, views=0,
               crops={"after": crop(tmp_path, "2_after", "red"), "before": None}),
        change(3, "unverified", before=side(2, [-2, 0, 3]), seen_in="before", conf=0.45,
               views=0, crops={"before": crop(tmp_path, "3_before", "grey")}),
        change(4, "added", after=side(2, [5, 0, 2]), conf=0.8,
               crops={"after": crop(tmp_path, "4_after", "green"),
                      "before": crop(tmp_path, "4_before", "grey")}),
        change(5, "removed", before=side(3, [10, 0, 2]), conf=0.7, views=7,
               crops={"before": crop(tmp_path, "5_before", "green"),
                      "after": crop(tmp_path, "5_after", "grey")}),
    ])
    return tmp_path


# ------------------------------------------------------------------ JSON and the fake


def test_extract_json_variants():
    assert extract_json('{"label": "chair"}') == {"label": "chair"}
    assert extract_json('Here: ```json\n{"a": {"b": 1}}\n``` done') == {"a": {"b": 1}}
    assert extract_json('noise {not json} then {"verdict": "rejected"}') == {"verdict": "rejected"}
    # broken JSON: single quotes, trailing comma, truncated reply -> regex fallback
    assert extract_json("{'verdict': 'confirmed', 'reason': \"it's gone\",}") == \
        {"verdict": "confirmed", "reason": "it's gone"}
    assert extract_json('{"label": "lamp", "attributes": "white', ) == {"label": "lamp"}
    assert extract_json('{"n": 3, "ok": true, "x": null') == {"n": 3, "ok": True, "x": None}
    assert extract_json("no json at all") is None


def test_ask_json_retries_once_and_logs(tmp_path):
    vlm = FakeVLM(["garbage", '{"verdict": "unclear"}'])
    vlm.open_log(tmp_path / "log.jsonl")
    image = Image.new("RGB", (2000, 1000))
    assert vlm.ask_json([image], "Question?", "t", ["crops/x.jpg"]) == {"verdict": "unclear"}
    assert vlm.calls == 2 and vlm.prompts[1].endswith(JSON_RETRY)
    assert vlm.image_sizes[0] == [(768, 384)]  # long side resized to image_max_side
    lines = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [x["tag"] for x in lines] == ["t", "t:retry"]
    assert lines[0]["images"] == ["crops/x.jpg"] and lines[0]["answer"] == "garbage"
    assert FakeVLM(["still garbage"]).ask_json([], "Q") is None


def test_label_normalisation():
    assert normalise_label("The Black Leather Bag.") == "black leather bag"
    assert normalise_label("a pair of shoes on the floor") == "pair of shoes"
    assert normalise_label(None) == "object" and normalise_label("  ") == "object"
    assert head_noun("curtains") == "curtain" and head_noun("glass") == "glass"
    assert head_noun("pastries") == "pastry"
    assert labels_match("left curtain", "curtains") and labels_match("bag", "bag")
    assert not labels_match("wall", "wall")  # parts of the room are never re-paired
    assert not labels_match("door frame", "white door frame")
    assert labels_match("beige duvet cover", "quilt")  # synonym group
    assert labels_match("picture frame", "frame") and labels_match("jacket", "coat")
    assert not labels_match("chair", "table") and not labels_match("object", "object")


# ------------------------------------------------------------------ stage logic


def test_naming_judge_and_repair(split_scene):
    vlm = FakeVLM(responder)
    doc = run_semantics(split_scene, load_config(), vlm)
    by_id = {c["id"]: c for c in doc["changes"]}
    assert [c["type"] for c in doc["changes"]] == ["moved", "removed", "added"]

    moved = by_id[1]  # removed bag + after-only bag 1 m away -> one moved change
    assert moved["label"] == "red bag" and moved["attributes"] == "red"
    assert moved["evidence"]["merged_from"] == [1, 2] and 2 not in by_id
    assert moved["before"]["centroid"] == [0, 0, 2] and moved["after"]["centroid"] == [1, 0, 2]
    assert moved["transform"]["translation_m"] == pytest.approx(1.0)
    assert moved["transform"]["rotation_deg"] is None
    assert moved["confidence"] == pytest.approx((0.9 + 0.4) / 2, abs=1e-3)
    assert moved["motion_kind"] == "rigid" and moved["evidence"]["vlm_verdict"] == "confirmed"
    assert moved["crops"]["before"] == "crops/1_before.jpg"
    assert moved["crops"]["after"] == "crops/2_after.jpg"
    assert moved["crops"]["before_place"] == "crops/1_after.jpg"
    # the re-pairing check uses the strict same-object prompt and also serves as the verdict
    same = [p for p in vlm.prompts if "same physical object" in p]
    assert len(same) == 1 and "already matched" not in same[0]

    assert by_id[5]["label"] == "hairbrush"  # 5 m from the hairbrush 4: too far to re-pair
    assert by_id[5]["evidence"]["vlm_verdict"] == "confirmed" and by_id[5]["confidence"] == 0.7
    assert by_id[4]["evidence"]["vlm_verdict"] == "unclear"
    assert by_id[4]["confidence"] == pytest.approx(0.8 * 0.7)  # unclear scales confidence

    assert [c["id"] for c in doc["rejected"]] == [3]  # the wall
    rejected = doc["rejected"][0]
    assert rejected["label"] == "wall" and rejected["evidence"]["vlm_reason"] == "only a plain wall"
    assert doc["summary"].startswith("3 changes: 1 moved, 1 added, 1 removed; 0 unverified. A red")
    for c in doc["changes"]:
        assert "viewpoint" in c["description"] and c["label"] in c["description"]

    meta = json.loads((split_scene / "semantics" / "meta.json").read_text())
    assert meta["counts_in"] == {"removed": 2, "unverified": 2, "added": 1}
    assert meta["counts_out"] == {"moved": 1, "removed": 1, "added": 1}
    assert meta["merges"][0]["ids"] == [1, 2] and meta["merges"][0]["verdict"] == "confirmed"
    assert meta["vlm_calls"] == vlm.calls and meta["template_kept"] == 3
    log = (split_scene / "semantics" / "log.jsonl").read_text().splitlines()
    assert len(log) == vlm.calls
    assert json.loads((split_scene / "changes.json").read_text()) == doc

    # re-running starts again from the M4 backup and gives the same result
    again = run_semantics(split_scene, load_config(), FakeVLM(responder))
    assert again == doc
    assert (split_scene / "semantics" / "changes_m4.json").is_file()


def test_judge_and_repair_can_be_disabled(split_scene):
    cfg = load_config(overrides=["semantics.enable_judge=false",
                                 "semantics.enable_repair=false"])
    doc = run_semantics(split_scene, cfg, FakeVLM(responder))
    assert len(doc["changes"]) == 5 and doc["rejected"] == []
    assert all(c["evidence"]["vlm_verdict"] is None for c in doc["changes"])
    assert doc["changes"][0]["label"] == "red bag"


def test_unparsable_judge_is_unclear(split_scene):
    def lazy(images, prompt):
        return responder(images, prompt) if '"label"' in prompt else "no idea"

    doc = run_semantics(split_scene, load_config(overrides=["semantics.enable_repair=false"]),
                        FakeVLM(lazy))
    # the region named "wall" cannot be confirmed as an object, whatever the judge says
    assert [c["id"] for c in doc["rejected"]] == [3]
    assert doc["rejected"][0]["evidence"]["vlm_reason"].startswith("named 'wall'")
    for c in doc["changes"]:
        assert c["evidence"]["vlm_verdict"] == "unclear"
    removed = next(c for c in doc["changes"] if c["id"] == 1)
    assert removed["confidence"] == pytest.approx(0.9 * 0.7)


def quilt_scene(root, gap):
    moved = change(1, "moved", before=side(1, [0, 0, 2], 0.2), after=side(1, [1.2, 0.2, 2.2], 0.2),
                   crops={k: crop(root, f"1_{k}", "red" if k in ("before", "after") else "grey")
                          for k in ("before", "after", "before_place", "after_place")})
    moved["transform"] = {"translation_m": 1.24, "rotation_deg": 15.0, "R": np.eye(3).tolist(),
                          "t": [1.2, 0.2, 0.2], "icp_fitness": 0.5, "icp_rmse": 0.01}
    moved["evidence"].update(observing_views_before_place=9, observing_views_after_place=6)
    part = side(2, [1.4 + gap + 0.5, 0.2, 2.2], 0.5, n=1500)
    added = change(2, "added", after=part, conf=0.6,
                   crops={"after": crop(root, "2_after", "red"),
                          "before": crop(root, "2_before", "grey")})
    write_doc(root, [moved, added])


@pytest.mark.parametrize("gap, merged", [(0.5, True), (1.5, False)])
def test_moved_absorbs_nearby_part(tmp_path, gap, merged):
    quilt_scene(tmp_path, gap)
    vlm = FakeVLM(responder)
    doc = run_semantics(tmp_path, load_config(), vlm)
    # the M4 moved pair is judged with the variant that knows about the geometric match
    assert sum("already matched by 3D geometry" in p for p in vlm.prompts) == 1
    if not merged:
        assert [c["type"] for c in doc["changes"]] == ["moved", "added"]
        return
    assert len(doc["changes"]) == 1
    m = doc["changes"][0]
    assert m["type"] == "moved" and m["evidence"]["merged_from"] == [1, 2]
    lo, hi = np.array(m["after"]["aabb"])
    assert np.allclose(lo, [1.0, -0.3, 1.7]) and np.allclose(hi, [2.9, 0.7, 2.7])
    assert np.allclose(m["after"]["extents"], hi - lo)
    assert m["after"]["instance_ids"] == [1, 2] and m["after"]["n_points"] == 2000
    obb = m["after"]["obb"]  # the oriented box encloses both parts
    corners = np.array([[x, y, z] for x in (1.0, 1.4, 1.9, 2.9) for y in (-0.3, 0.7)
                        for z in (1.7, 2.7)])
    local = (corners - obb["center"]) @ rotation_y(obb["yaw_deg"])
    assert np.all(np.abs(local) <= np.array(obb["extents"]) / 2 + 1e-3)
    assert obb["extents"][1] == pytest.approx(1.0)
    assert m["crops"]["after_part"] == "crops/2_after.jpg"
    assert m["transform"]["rotation_deg"] is None
    assert m["transform"]["translation_m"] == pytest.approx(
        np.linalg.norm(np.subtract(m["after"]["centroid"], [0, 0, 2])), abs=1e-3)
    assert "merges candidates 1, 2" in m["description"]


def test_rejected_moved_pair_is_split_into_removed_and_added(tmp_path):
    """The judge rejects an M4 moved pair (a red bag before, a green hairbrush after): both
    sides are kept, as a removed and an added change, named again, confidence x 0.6,
    verdict "unclear"."""
    moved = change(3, "moved", before=side(4, [0, 0, 2]), after=side(5, [1, 0, 2]), conf=0.8,
                   crops={"before": crop(tmp_path, "3_before", "red"),
                          "after": crop(tmp_path, "3_after", "green"),
                          "before_place": crop(tmp_path, "3_before_place", "grey"),
                          "after_place": crop(tmp_path, "3_after_place", "grey")})
    moved["transform"] = {"translation_m": 1.0, "rotation_deg": None, "R": np.eye(3).tolist(),
                          "t": [1.0, 0.0, 0.0], "icp_fitness": 0.2, "icp_rmse": 0.02}
    moved["evidence"].update(observing_views_before_place=6, observing_views_after_place=2)
    other = change(7, "removed", before=side(6, [-3, 0, 2]), conf=0.9,
                   crops={"before": crop(tmp_path, "7_before", "red"),
                          "after": crop(tmp_path, "7_after", "grey")})
    write_doc(tmp_path, [moved, other])
    (tmp_path / "instances").mkdir()  # the before instance's own evidence is used if present
    record = {"id": 4, "mask_score_mean": 0.85, "view_consistency": 0.7,
              "evidence": {"free_rays": 321, "free_frames_mean": 4.0, "observing_views": 6,
                           "occ_points": 800}}
    (tmp_path / "instances" / "before.json").write_text(json.dumps({"instances": [record]}))
    doc = run_semantics(tmp_path, load_config(), FakeVLM(responder))

    assert doc["rejected"] == []
    by_id = {c["id"]: c for c in doc["changes"]}
    assert sorted(by_id) == [3, 7, 8]  # the added half gets the next free id
    removed, added = by_id[3], by_id[8]
    assert removed["type"] == "removed" and added["type"] == "added"
    assert removed["before"] == moved["before"] and removed["after"] is None
    assert added["after"] == moved["after"] and added["before"] is None
    assert removed["label"] == "red bag" and added["label"] == "hairbrush"  # named again
    for c in (removed, added):
        assert c["confidence"] == pytest.approx(0.8 * 0.6)
        assert c["evidence"]["vlm_verdict"] == "unclear" and c["evidence"]["split_from"] == 3
        assert c["evidence"]["vlm_reason"] == "appearance judge did not confirm the match: colours"
        assert c["transform"] is None and c["motion_kind"] is None
        assert "viewpoint" in c["description"]
    assert removed["crops"] == {"before": "crops/3_before.jpg", "after": "crops/3_before_place.jpg"}
    assert added["crops"] == {"after": "crops/3_after.jpg", "before": "crops/3_after_place.jpg"}
    assert removed["evidence"]["observing_views"] == 6 and added["evidence"]["observing_views"] == 2
    assert removed["evidence"]["free_rays"] == 321 and removed["evidence"]["mask_score"] == 0.85
    assert "free_rays" not in added["evidence"]  # no instance record: per-place evidence
    meta = json.loads((tmp_path / "semantics" / "meta.json").read_text())
    assert meta["splits"] == [{"id": 3, "label": "red bag", "into": [3, 8],
                               "labels": ["red bag", "hairbrush"], "reason": "colours"}]
    assert meta["verdicts"] == {"3": "unclear", "7": "confirmed", "8": "unclear"}


def test_split_half_is_re_paired_with_another_change(tmp_path):
    """M4 paired the wrong objects (a red bag before, a green hairbrush after) and missed
    the removed hairbrush 7: after the split, the added hairbrush half pairs with it."""
    moved = change(3, "moved", before=side(4, [0, 0, 2]), after=side(5, [1, 0, 2]), conf=0.8,
                   crops={"before": crop(tmp_path, "3_before", "red"),
                          "after": crop(tmp_path, "3_after", "green"),
                          "before_place": crop(tmp_path, "3_before_place", "grey"),
                          "after_place": crop(tmp_path, "3_after_place", "grey")})
    moved["evidence"].update(observing_views_before_place=6, observing_views_after_place=2)
    brush = change(7, "removed", before=side(6, [0.5, 0, 2]), conf=0.9,
                   crops={"before": crop(tmp_path, "7_before", "green"),
                          "after": crop(tmp_path, "7_after", "grey")})
    write_doc(tmp_path, [moved, brush])
    vlm = FakeVLM(responder)
    doc = run_semantics(tmp_path, load_config(), vlm)

    assert [(c["id"], c["type"], c["label"]) for c in doc["changes"]] == [
        (7, "moved", "hairbrush"), (3, "removed", "red bag")]
    m = doc["changes"][0]
    assert m["evidence"]["merged_from"] == [7, 8] and m["evidence"]["vlm_verdict"] == "confirmed"
    assert m["after"] == moved["after"] and m["motion_kind"] == "rigid"
    assert m["confidence"] == pytest.approx((0.9 + 0.8 * 0.6) / 2, abs=1e-3)
    meta = json.loads((tmp_path / "semantics" / "meta.json").read_text())
    assert [x["ids"] for x in meta["merges"]] == [[7, 8]]  # never the two halves 3 + 8
    assert sum("same physical object" in p and "already matched" not in p
               for p in vlm.prompts) == 1  # one strict same-object check: 7 + 8


CHAIR_PARTS = {"red": "Office Chair", "green": "office chair base", "grey": "wall"}


def chair_scene(root):
    """Two removed fragments of one chair (upper part 1, base 2, 5 cm below it), a second
    removed chair 1 m away (3) and an added chair (4); instance points on disk."""
    rng = np.random.default_rng(0)
    pts = {1: rng.uniform([-0.25, 0.40, 1.75], [0.25, 1.00, 2.25], (3000, 3)),
           2: rng.uniform([-0.30, 0.00, 1.70], [0.30, 0.35, 2.30], (800, 3)),
           3: rng.uniform([1.30, 0.00, 1.75], [1.80, 1.00, 2.25], (3000, 3))}

    def side_of(iid):
        p = pts[iid]
        return {"instance_id": iid, "centroid": p.mean(0).tolist(),
                "aabb": [p.min(0).tolist(), p.max(0).tolist()], "extents": np.ptp(p, 0).tolist(),
                "obb": floor_obb(p), "n_points": len(p), "blob_ids": [10 + iid]}

    write_doc(root, [
        change(1, "removed", before=side_of(1), conf=0.56,
               crops={"before": crop(root, "1_before", "red"),
                      "after": crop(root, "1_after", "grey")}),
        change(2, "removed", before=side_of(2), conf=0.93, views=8,
               crops={"before": crop(root, "2_before", "green"),
                      "after": crop(root, "2_after", "grey")}),
        change(3, "removed", before=side_of(3), conf=0.8,
               crops={"before": crop(root, "3_before", "red"),
                      "after": crop(root, "3_after", "grey")}),
        change(4, "added", after=side(1, [5, 0.5, 2], 0.25), conf=0.7,
               crops={"after": crop(root, "4_after", "green"),
                      "before": crop(root, "4_before", "grey")}),
    ])
    (root / "instances").mkdir()
    np.savez_compressed(root / "instances" / "before_points.npz",
                        points=np.concatenate(list(pts.values())).astype(np.float32),
                        colors=np.zeros((sum(map(len, pts.values())), 3), np.uint8),
                        instance=np.concatenate([np.full(len(p), i) for i, p in pts.items()]))
    return pts


def chair_responder(parts_answer):
    def answer(images, prompt):
        if '"label"' in prompt and "parts of one physical object" not in prompt:
            label = CHAIR_PARTS[colour(images[0])]
            return json.dumps({"label": label, "attributes": "grey fabric",
                               "description": "A chair part."})
        if "parts of one physical object" in prompt:
            return parts_answer
        return responder(images, prompt)
    return answer


def test_fragments_of_one_object_are_merged(tmp_path):
    pts = chair_scene(tmp_path)
    vlm = FakeVLM(chair_responder(
        '{"reason": "seat and base of one chair", "verdict": "confirmed", '
        '"label": "an office chair"}'))
    doc = run_semantics(tmp_path, load_config(), vlm)

    asked = [p for p in vlm.prompts if "parts of one physical object" in p]
    assert len(asked) == 1  # only 1 + 2: chair 3 is 1 m away, the added chair 4 is alone
    assert "probably a office chair" in asked[0]
    by_id = {c["id"]: c for c in doc["changes"]}
    assert sorted(by_id) == [1, 3, 4] and doc["rejected"] == []
    m = by_id[1]  # the fragment with more points keeps its id
    assert m["type"] == "removed" and m["label"] == "office chair"
    assert m["confidence"] == 0.93  # max of both, then confirmed by the judge
    assert m["evidence"]["merged_from"] == [1, 2]
    assert m["evidence"]["merged_types"] == ["removed", "removed"]
    assert m["evidence"]["merged_part_reason"] == "seat and base of one chair"
    assert m["evidence"]["fragment_gap_m"] == pytest.approx(0.05, abs=0.01)
    assert m["evidence"]["free_rays"] == 200 and m["evidence"]["observing_views"] == 8
    assert m["evidence"]["vlm_verdict"] == "confirmed"
    union = np.concatenate([pts[1], pts[2]])
    b = m["before"]
    assert b["instance_ids"] == [1, 2] and b["n_points"] == 3800 and b["blob_ids"] == [11, 12]
    assert np.allclose(b["centroid"], union.mean(0), atol=1e-4)
    assert np.allclose(b["aabb"], [union.min(0), union.max(0)], atol=1e-4)
    assert b["obb"] == floor_obb(union) and b["obb"]["extents"][1] == pytest.approx(1.0, abs=0.01)
    assert m["crops"] == {"before": "crops/1_before.jpg", "after": "crops/1_after.jpg",
                          "before_part": "crops/2_before.jpg"}
    assert m["description"].startswith("The office chair (grey fabric; about 1.0 x 0.6 x 0.6 m)")
    frame = SceneFrame(up=np.array([0.0, 1.0, 0.0]), right=np.array([1.0, 0.0, 0.0]),
                       forward=np.array([0.0, 0.0, 1.0]), floor=0.0,
                       lo=np.array([-3.0, 0.0]), hi=np.array([3.0, 6.0]))
    text = template_description(m, frame, {"before": {1: pts[1], 2: pts[2]}})
    assert "which was on the floor" in text  # the base's points count for the height
    assert "0.4 m above the floor" in template_description(by_id[3] | {"before": {
        **m["before"], "instance_ids": [1]}}, frame, {"before": {1: pts[1]}})
    assert by_id[3]["label"] == "office chair" and "merged_from" not in by_id[3]["evidence"]
    meta = json.loads((tmp_path / "semantics" / "meta.json").read_text())
    assert meta["merges"] == [{"kind": "removed+removed", "ids": [1, 2],
                               "labels": ["office chair", "office chair base"],
                               "obb_gap_m": m["evidence"]["fragment_gap_m"],
                               "verdict": "confirmed", "reason": "seat and base of one chair",
                               "label": "office chair"}]
    assert meta["counts_out"] == {"removed": 2, "added": 1}


def test_fragments_stay_apart_when_the_vlm_refuses(tmp_path):
    chair_scene(tmp_path)
    vlm = FakeVLM(chair_responder(
        '{"reason": "two different chairs", "verdict": "rejected", "label": "chairs"}'))
    doc = run_semantics(tmp_path, load_config(), vlm)
    by_id = {c["id"]: c for c in doc["changes"]}
    assert sorted(by_id) == [1, 2, 3, 4]
    assert by_id[2]["label"] == "office chair base" and by_id[1]["before"]["instance_id"] == 1
    assert "instance_ids" not in by_id[1]["before"] and by_id[1]["confidence"] == 0.56
    meta = json.loads((tmp_path / "semantics" / "meta.json").read_text())
    assert [(x["ids"], x["verdict"]) for x in meta["merges"]] == [([1, 2], "rejected")]
    # the switch turns the step off: no question asked
    vlm = FakeVLM(chair_responder('{"verdict": "confirmed"}'))
    doc = run_semantics(tmp_path, load_config(overrides=["semantics.enable_fragment_merge=false"]),
                        vlm)
    assert len(doc["changes"]) == 4
    assert not any("parts of one physical object" in p for p in vlm.prompts)


def test_obb_gap():
    a = {"center": [0.0, 0.5, 0.0], "extents": [1.0, 1.0, 1.0], "yaw_deg": 0.0}
    assert obb_gap(a, a) == 0.0
    assert obb_gap(a, {**a, "center": [2.0, 0.5, 0.0]}) == pytest.approx(1.0)
    turned = {**a, "center": [2.0, 0.5, 0.0], "yaw_deg": 45.0}  # a corner points at a
    assert obb_gap(a, turned) == pytest.approx(1.5 - np.sqrt(0.5))
    above = {**a, "center": [2.0, 2.5, 0.0]}
    assert obb_gap(a, above) == pytest.approx(np.sqrt(2.0))
    assert obb_gap(a, {**a, "center": [0.2, 1.6, 0.1]}) == pytest.approx(0.1)


# ------------------------------------------------------------------ descriptions


def test_template_description_geometry():
    frame = SceneFrame(up=np.array([0.0, -1.0, 0.0]), right=np.array([1.0, 0.0, 0.0]),
                       forward=np.array([0.0, 0.0, 1.0]), floor=-1.0,
                       lo=np.array([-3.0, 0.0]), hi=np.array([3.0, 6.0]))
    c = change(1, "removed", before=side(7, [-2.5, 0.8, 5.5], 0.1), views=7)
    c["label"], c["attributes"] = "backpack", "black nylon"
    floor_pts = np.array([[-2.5, 0.95, 5.5], [-2.5, 0.65, 5.5]] * 20)  # bottom at y = 0.95
    text = template_description(c, frame, {"before": {7: floor_pts}})
    assert text.startswith("The backpack (black nylon; about 20 x 20 x 20 cm), which was on the "
                           "floor in the far left part of the room, was removed")
    assert "from 7 viewpoints and saw it empty" in text
    c = change(2, "unverified", after=side(3, [2.5, -0.5, 0.5], 0.4), seen_in="after")
    c["label"] = "pajamas"
    text = template_description(c, frame, {})
    assert text.startswith("The pajamas (about 0.8 x 0.8 x 0.8 m), 1.1 m above the floor in "
                           "the near right part of the room, was seen only in the after")
    m = change(3, "moved", before=side(1, [0, 0.8, 3]), after=side(1, [0.9, 0.8, 3]))
    m["label"], m["motion_kind"] = "ladder", "rigid"
    m["transform"] = {"translation_m": 0.93, "rotation_deg": 31.0}
    m["evidence"].update(observing_views_before_place=6, observing_views_after_place=1)
    text = template_description(m, frame, {})
    assert "moved 0.9 m within the middle of the room (on the floor both times) and turned by " \
        "about 31 degrees;" in text
    assert "from 6 viewpoints" in text and "from 1 viewpoint." in text


def test_polish_keeps_only_sentences_with_all_numbers():
    sentences = ["The bag moved 0.8 m; seen from 7 viewpoints.", "A cup appeared (2 viewpoints)."]
    answer = json.dumps({"sentences": ["Seen from 7 viewpoints, the bag moved 0.8 m.",
                                       "A mug showed up, seen twice."]})
    out = polish(FakeVLM([answer]), sentences, ["bag", "cup"])
    assert out == ["Seen from 7 viewpoints, the bag moved 0.8 m.", None]
    assert polish(FakeVLM(['{"sentences": ["only one"]}']), sentences, ["bag", "cup"]) == \
        [None, None]
    assert numbers_preserved("moved 0.8 m", "The chair moved 0.8 m.", "chair")
    assert not numbers_preserved("moved 0.8 m", "The chair moved 0.80 m.", "chair")
    assert not numbers_preserved("moved 0.8 m", "It moved 0.8 m.", "chair")


# ------------------------------------------------------------------ CLI on the synthetic scene


def test_cli_describe_and_viewer(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import roomdiff.models
    import roomdiff.vlm
    from roomdiff.cli import app
    from roomdiff.models import FakeFeaturizer, FakeSegmenter
    from tests.test_changes import SETTINGS, make_scene

    labels = make_scene(tmp_path)
    monkeypatch.setattr(roomdiff.models, "load_models",
                        lambda params: (FakeSegmenter(labels), FakeFeaturizer()))
    fake = FakeVLM()
    monkeypatch.setattr(roomdiff.vlm, "load_vlm", lambda params: fake)
    runner = CliRunner()
    sets = [x for s in SETTINGS for x in ("-s", s)]
    for command in ("visibility", "instances", "changes", "describe"):
        result = runner.invoke(app, [command, "--out", str(tmp_path), *sets])
        assert result.exit_code == 0, result.output
    assert "[semantics]" in result.output and "[report]" in result.output
    doc = json.loads((tmp_path / "changes.json").read_text())
    assert set(doc) == {"scene", "changes", "rejected", "summary"}
    # the fake names everything "box" and calls every pair the same object, so the
    # removed blue box and the added green box (0.8 m apart) are re-paired into a move
    assert doc["summary"].startswith("2 changes: 2 moved, 0 added, 0 removed; ")
    assert doc["summary"].endswith("A scripted summary of the scene change.")
    for c in doc["changes"]:
        assert c["label"] == "box" and c["evidence"]["vlm_verdict"] == "confirmed"
        assert ("viewpoint" if c["type"] != "unverified" else "never looked") in c["description"]
    moved = next(c for c in doc["changes"] if c["type"] == "moved")
    assert moved["motion_kind"] == "rigid"
    assert fake.calls > len(doc["changes"])
    assert all(max(size) <= 768 for sizes in fake.image_sizes for size in sizes)
    report = (tmp_path / "report.md").read_text()
    assert "`MOVED` box" in report and "](crops/" in report
    assert (tmp_path / "semantics" / "log.jsonl").stat().st_size > 0

    result = runner.invoke(app, ["viewer", "--out", str(tmp_path), "--target-points", "5000"])
    assert result.exit_code == 0, result.output
    viewer_doc = json.loads((tmp_path / "viewer" / "changes.json").read_text())
    assert viewer_doc["summary"] == doc["summary"]
    assert (tmp_path / "viewer" / "index.html").is_file()
