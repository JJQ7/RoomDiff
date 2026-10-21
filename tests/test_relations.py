"""Tests for roomdiff.relations: supports, covers, on / under / beside and the phrases, on
synthetic clouds in a levelled frame (+Y up, floor at y = 0; CPU only)."""

import json

import numpy as np
import pytest

from roomdiff.config import load_config
from roomdiff.geometry import obb_footprint, rotation_y
from roomdiff.recon import write_recon
from roomdiff.relations import (
    Place,
    add_relations,
    box_coords,
    change_text,
    compute_relations,
    find_support,
    horizontal_patch,
    relation_text,
    run_relations,
)

STEP = 0.01  # m, spacing of the synthetic surface points
PARAMS = dict(load_config()["relations"])
SLAB_Y = 0.70  # table top
SLAB = ((-1.0, -0.5), (-0.2, 0.5))  # (x0, z0), (x1, z1)


def plane(x0: float, z0: float, x1: float, z1: float, y: float,
          holes: tuple = ()) -> np.ndarray:
    """Points of a horizontal rectangle at height `y`, minus the footprints in `holes`
    (((x0, z0), (x1, z1)) each: the object standing there hides the surface)."""
    x, z = np.meshgrid(np.arange(x0, x1 + 1e-9, STEP), np.arange(z0, z1 + 1e-9, STEP))
    pts = np.c_[x.ravel(), np.full(x.size, y), z.ravel()]
    for (hx0, hz0), (hx1, hz1) in holes:
        inside = (pts[:, 0] > hx0) & (pts[:, 0] < hx1) & (pts[:, 2] > hz0) & (pts[:, 2] < hz1)
        pts = pts[~inside]
    return pts


def box_points(lo, hi) -> np.ndarray:
    """Points on the six faces of the box [lo, hi]."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    faces = []
    for axis in range(3):
        a, b = [k for k in range(3) if k != axis]
        u, v = np.meshgrid(np.arange(lo[a], hi[a] + 1e-9, STEP),
                           np.arange(lo[b], hi[b] + 1e-9, STEP))
        for value in (lo[axis], hi[axis]):
            face = np.empty((u.size, 3))
            face[:, axis], face[:, a], face[:, b] = value, u.ravel(), v.ravel()
            faces.append(face)
    return np.concatenate(faces)


def footprint(lo, hi):
    return (lo[0], lo[2]), (hi[0], hi[2])


def side(instance_id: int, lo, hi) -> dict:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return {"instance_id": instance_id, "centroid": ((lo + hi) / 2).tolist(),
            "aabb": [lo.tolist(), hi.tolist()], "extents": (hi - lo).tolist(),
            "obb": {"center": ((lo + hi) / 2).tolist(), "extents": (hi - lo).tolist(),
                    "yaw_deg": 0.0}}


def change(cid: int, kind: str, label: str, before=None, after=None) -> dict:
    tf = None
    if before and after:
        t = np.subtract(after["centroid"], before["centroid"])
        tf = {"translation_m": round(float(np.linalg.norm(t)), 4)}
    return {"id": cid, "type": kind, "label": label, "confidence": 0.9, "before": before,
            "after": after, "transform": tf, "evidence": {}}


# before: a cup on the table top, a bag on the floor under it, two boxes side by side,
# a lamp hanging in the air, a can on the table top; after: the can on the floor, a
# parcel on the floor with a phone on it.
CUP = ((-0.70, SLAB_Y, -0.10), (-0.55, SLAB_Y + 0.15, 0.05))
BAG = ((-0.80, 0.0, 0.20), (-0.50, 0.30, 0.45))
LEFT = ((0.50, 0.0, 0.00), (0.80, 0.30, 0.30))
RIGHT = ((0.85, 0.0, 0.00), (1.15, 0.30, 0.30))
LAMP = ((1.50, 1.40, -0.60), (1.70, 1.60, -0.40))
CAN_B = ((-0.40, SLAB_Y, -0.35), (-0.28, SLAB_Y + 0.15, -0.23))
CAN_A = ((0.00, 0.0, 1.00), (0.12, 0.15, 1.12))
PARCEL = ((1.00, 0.0, -1.00), (1.40, 0.30, -0.60))
PHONE = ((1.10, 0.30, -0.90), (1.20, 0.32, -0.80))
BEFORE = {1: CUP, 2: BAG, 3: LEFT, 4: RIGHT, 5: LAMP, 6: CAN_B}
AFTER = {6: CAN_A, 7: PARCEL, 8: PHONE}


def scene() -> tuple[list[dict], dict, dict]:
    """Changes, recording clouds and instance points of the synthetic room."""
    points = {"before": {i: box_points(*b) for i, b in BEFORE.items()},
              "after": {i: box_points(*b) for i, b in AFTER.items()}}
    on_floor_b = [footprint(*b) for b in (BAG, LEFT, RIGHT)]
    on_slab_b = [footprint(*b) for b in (CUP, CAN_B)]
    clouds = {
        "before": np.concatenate([plane(-2, -2, 2, 2, 0.0, tuple(on_floor_b)),
                                  plane(*SLAB[0], *SLAB[1], SLAB_Y, tuple(on_slab_b)),
                                  *points["before"].values()]),
        "after": np.concatenate([plane(-2, -2, 2, 2, 0.0, (footprint(*CAN_A),
                                                           footprint(*PARCEL))),
                                 plane(*SLAB[0], *SLAB[1], SLAB_Y),
                                 *points["after"].values()]),
    }
    changes = [
        change(6, "moved", "can", before=side(6, *CAN_B), after=side(6, *CAN_A)),
        change(1, "removed", "cup", before=side(1, *CUP)),
        change(2, "removed", "bag", before=side(2, *BAG)),
        change(3, "removed", "left box", before=side(3, *LEFT)),
        change(4, "removed", "right box", before=side(4, *RIGHT)),
        change(7, "added", "parcel", after=side(7, *PARCEL)),
        change(8, "added", "phone", after=side(8, *PHONE)),
        change(5, "unverified", "lamp", before=side(5, *LAMP)),
    ]
    return changes, clouds, points


@pytest.fixture(scope="module")
def related():
    changes, clouds, points = scene()
    counts = compute_relations(changes, clouds, points, PARAMS)
    return {c["id"]: c for c in changes}, counts


def test_object_on_a_slab(related):
    by_id, _ = related
    support = by_id[1]["relations"]["before"]["support"]
    assert support["kind"] == "surface" and support["height"] == pytest.approx(SLAB_Y, abs=0.01)
    assert support["coverage"] > 0.5 and abs(support["gap"]) < 0.02
    assert by_id[1]["relations"]["after"] is None
    assert by_id[1]["relations"]["text"] == "on a surface 0.70 m above the floor"


def test_object_on_the_floor(related):
    by_id, counts = related
    support = by_id[3]["relations"]["before"]["support"]
    assert support["kind"] == "floor" and abs(support["height"]) < 0.01
    assert by_id[3]["relations"]["before"]["cover"] is None
    assert counts["floor"] >= 4 and counts["surface"] >= 2


def test_object_under_a_slab(related):
    by_id, _ = related
    place = by_id[2]["relations"]["before"]
    assert place["support"]["kind"] == "floor"
    assert place["cover"]["height"] == pytest.approx(SLAB_Y, abs=0.01)
    assert place["cover"]["coverage"] > 0.9
    assert by_id[2]["relations"]["text"] == "on the floor, under a surface 0.70 m above the floor"


def test_two_objects_side_by_side(related):
    by_id, _ = related
    assert by_id[3]["relations"]["before"]["objects"] == [
        {"relation": "beside", "other_change_id": 4}]
    assert by_id[4]["relations"]["before"]["objects"] == [
        {"relation": "beside", "other_change_id": 3}]
    assert by_id[3]["relations"]["text"] == "on the floor, beside the right box"


def test_object_on_another_change(related):
    by_id, _ = related
    phone = by_id[8]["relations"]["after"]
    assert phone["objects"] == [{"relation": "on", "other_change_id": 7}]
    assert phone["support"]["kind"] == "surface"
    assert phone["support"]["height"] == pytest.approx(0.30, abs=0.01)
    assert by_id[8]["relations"]["text"] == "on the parcel"
    # the parcel is below the phone, not under it (they touch) and not beside it
    assert by_id[7]["relations"]["after"]["objects"] == []
    assert by_id[7]["relations"]["text"] == "on the floor"


def test_hanging_object_has_no_support(related):
    by_id, counts = related
    assert by_id[5]["relations"]["before"]["support"]["kind"] == "none"
    assert by_id[5]["relations"]["text"] == ""
    assert counts["none"] == 1


def test_moved_from_a_slab_to_the_floor(related):
    by_id, _ = related
    rel = by_id[6]["relations"]
    assert rel["moved"]["from"] == {"kind": "surface", "height": pytest.approx(SLAB_Y, abs=0.01)}
    assert rel["moved"]["to"] == {"kind": "floor"}
    assert rel["moved"]["distance_m"] == by_id[6]["transform"]["translation_m"]
    assert rel["moved"]["horizontal_m"] == pytest.approx(np.hypot(0.40, 1.35), abs=0.01)
    assert rel["text"] == "from the surface at 0.70 m to the floor"


def test_text_without_a_known_floor():
    changes, clouds, points = scene()
    compute_relations(changes, clouds, points, PARAMS, floor_known=False)
    by_id = {c["id"]: c for c in changes}
    assert by_id[3]["relations"]["before"]["support"]["kind"] == "surface"
    assert by_id[1]["relations"]["text"] == "on a horizontal surface"
    assert by_id[2]["relations"]["text"] == "on a horizontal surface, under a horizontal surface"
    assert by_id[6]["relations"]["text"] == "from a surface to one 0.70 m lower"


def test_text_names_kept_changes_only(related):
    by_id, _ = related
    left = by_id[3]
    assert relation_text(left, {3: "left box"}) == "on the floor"  # the right box was rejected
    assert relation_text(left, {3: "object", 4: "object"}) == "on the floor, beside change 4"
    doc = {"changes": [left], "relations_info": {"floor_known": True}}
    assert change_text(doc, left) == "on the floor"
    assert relation_text({"type": "removed", "relations": None}, {}) == ""


def test_rejected_entries_get_support_only():
    changes, clouds, points = scene()
    kept, rejected = changes[:-2], changes[-2:]  # phone and lamp rejected
    compute_relations(kept, clouds, points, PARAMS, rejected)
    assert rejected[0]["relations"]["after"]["objects"] == []
    assert rejected[0]["relations"]["after"]["support"]["kind"] == "surface"
    parcel = next(c for c in kept if c["id"] == 7)
    assert parcel["relations"]["after"]["objects"] == []


def test_turned_box_footprint():
    """Box coordinates of a box turned about +Y: its footprint corners land on +-half
    its extents, and a turned object on the table top still finds it."""
    obb = {"center": [-0.6, SLAB_Y + 0.075, 0.0], "extents": [0.3, 0.15, 0.1], "yaw_deg": 30.0}
    uv = box_coords(obb, obb_footprint(obb))
    assert np.allclose(np.abs(uv), [0.15, 0.05], atol=1e-9)
    local = box_points((-0.15, -0.075, -0.05), (0.15, 0.075, 0.05))
    pts = local @ rotation_y(30.0).T + obb["center"]
    inside = np.abs(box_coords(obb, plane(*SLAB[0], *SLAB[1], SLAB_Y)[:, [0, 2]]))
    table = plane(*SLAB[0], *SLAB[1], SLAB_Y)[~np.all(inside < [0.15, 0.05], axis=1)]
    cloud = np.concatenate([plane(-2, -2, 2, 2, 0.0), table, pts])
    place = Place(1, "before", obb, SLAB_Y, SLAB_Y + 0.15, pts)
    support = find_support(cloud, place, PARAMS)
    assert support["kind"] == "surface" and support["height"] == pytest.approx(SLAB_Y, abs=0.01)
    assert support["coverage"] > 0.8


def test_horizontal_patch_needs_a_level_surface():
    rng = np.random.default_rng(0)
    n = 4000
    cells = rng.integers(0, 100, n)
    flat = rng.normal(0.5, 0.004, n)
    found = horizontal_patch(cells, flat, 100, 0.3, 0.7, True, 0.015, 0.25, 30, 0.5)
    assert found is not None and found["height"] == pytest.approx(0.5, abs=0.005)
    # a wall: every height in the band, in a line of cells
    wall_cells = rng.integers(0, 30, n)
    wall = rng.uniform(0.0, 1.0, n)
    assert horizontal_patch(wall_cells, wall, 100, 0.3, 0.7, True, 0.015, 0.25, 30, 0.5) is None
    # the same wall over all cells: dense in every window, but not a peak
    assert horizontal_patch(cells, wall, 100, 0.3, 0.7, True, 0.015, 0.25, 30, 0.5) is None
    # two levels: scanning down finds the upper one first
    two = np.where(np.arange(n) % 2 == 0, 0.6, 0.4) + rng.normal(0, 0.003, n)
    assert horizontal_patch(cells, two, 100, 0.3, 0.7, True, 0.015, 0.25, 30, 0.5)[
        "height"] == pytest.approx(0.6, abs=0.005)
    assert horizontal_patch(cells, two, 100, 0.3, 0.7, False, 0.015, 0.25, 30, 0.5)[
        "height"] == pytest.approx(0.4, abs=0.005)


# ------------------------------------------------------------------ document level


def write_meta(out, **meta):
    (out / "recon").mkdir(parents=True, exist_ok=True)
    (out / "recon" / "meta.json").write_text(json.dumps(meta))


def test_unlevelled_frame_gives_null_relations(tmp_path):
    write_meta(tmp_path, levelled=False)
    changes, _, _ = scene()
    doc = {"scene": {}, "changes": changes[:3], "rejected": changes[3:4], "summary": ""}
    info = add_relations(tmp_path, doc, load_config())
    assert info["levelled"] is False and "not levelled" in info["reason"]
    assert doc["relations_info"] == info
    assert all(c["relations"] is None for c in doc["changes"] + doc["rejected"])
    assert change_text(doc, doc["changes"][0]) == ""


def test_disabled_leaves_the_document_alone(tmp_path):
    write_meta(tmp_path, levelled=True)
    changes, _, _ = scene()
    doc = {"scene": {}, "changes": changes, "summary": ""}
    before = json.dumps(doc)
    cfg = load_config(overrides=["relations.enabled=false"])
    assert add_relations(tmp_path, doc, cfg) is None
    assert json.dumps(doc) == before


def render_down(cam: np.ndarray, boxes, f: float = 100.0, size: int = 160):
    """z-depth of a camera at `cam` looking straight down (-Y) at the floor, the table top
    and the tops of axis-aligned `boxes` (lo, hi): depth, K, cam2world."""
    K = np.array([[f, 0, size / 2 - 0.5], [0, f, size / 2 - 0.5], [0, 0, 1]], np.float32)
    T = np.eye(4)
    T[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # x right, y = world +z, z = world -y
    T[:3, 3] = cam
    v, u = np.mgrid[0:size, 0:size]
    dx, dz = (u - K[0, 2]) / f, (v - K[1, 2]) / f  # world x / z offset per metre of depth
    tops = [((-9, -9), (9, 9), 0.0), (*SLAB, SLAB_Y)] + [
        ((lo[0], lo[2]), (hi[0], hi[2]), hi[1]) for lo, hi in boxes]
    depth = np.full((size, size), np.inf)
    for (x0, z0), (x1, z1), y in tops:  # the highest surface hit along the ray wins
        z = cam[1] - y
        x, zz = cam[0] + z * dx, cam[2] + z * dz
        hit = (x >= x0) & (x <= x1) & (zz >= z0) & (zz <= z1) & (z < depth)
        depth = np.where(hit, z, depth)
    return depth.astype(np.float32), K, T


def test_cli_relations_on_a_run(tmp_path):
    """`roomdiff relations` on a run folder: the cloud comes from recon.npz (top-down
    views), the instance points from instances/<rec>_points.npz; the document only gains
    keys and the M5 report shows the phrase."""
    from typer.testing import CliRunner

    from roomdiff.cli import app

    data = {k: [] for k in ("rec", "files", "images", "cam2world", "K", "depth", "conf",
                            "valid")}
    for rec, boxes in ((0, [CUP]), (1, [])):
        for k, cam in enumerate(([-0.6, 2.0, 0.0], [-0.3, 2.2, -0.2], [-0.9, 1.9, 0.3])):
            depth, K, T = render_down(np.array(cam), boxes)
            data["rec"].append(rec)
            data["files"].append(f"{('before', 'after')[rec]}_{k:05d}.jpg")
            data["images"].append(np.zeros((*depth.shape, 3), np.uint8))
            data["cam2world"].append(T)
            data["K"].append(K)
            data["depth"].append(depth)
            data["conf"].append(np.full(depth.shape, 0.9))
            data["valid"].append(np.isfinite(depth))
    recon = {k: np.stack(v) for k, v in data.items()}
    recon["metric_scale"] = 1.0
    write_meta(tmp_path, levelled=True, floor_inliers_fraction=0.9,
               level={"method": "ransac floor plane"})
    write_recon(tmp_path / "recon" / "recon.npz", recon)
    (tmp_path / "instances").mkdir()
    cup = box_points(*CUP).astype(np.float32)
    np.savez(tmp_path / "instances" / "before_points.npz", points=cup,
             colors=np.zeros((len(cup), 3), np.uint8), instance=np.ones(len(cup), np.int32))
    doc = {"scene": {"units": "metres (approx.)"},
           "changes": [{**change(1, "removed", "cup", before=side(1, *CUP)),
                        "description": "The cup was removed.", "crops": {}}],
           "rejected": [], "summary": "1 change"}
    (tmp_path / "changes.json").write_text(json.dumps(doc, indent=2))

    result = CliRunner().invoke(app, ["relations", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    new = json.loads((tmp_path / "changes.json").read_text())
    assert new["relations_info"]["floor_known"] is True
    rel = new["changes"][0].pop("relations")
    assert rel["before"]["support"]["kind"] == "surface"
    assert rel["text"] == "on a surface 0.70 m above the floor"
    del new["relations_info"]
    assert new == doc  # only keys were added
    assert "**where** on a surface 0.70 m above the floor" in (tmp_path / "report.md").read_text()

    # a floor plane holding few of the lowest points may be a counter: no heights quoted
    write_meta(tmp_path, levelled=True, floor_inliers_fraction=0.09,
               level={"method": "ransac floor plane"})
    run_relations(tmp_path, load_config())
    new = json.loads((tmp_path / "changes.json").read_text())
    assert new["relations_info"]["floor_known"] is False
    assert new["changes"][0]["relations"]["text"] == "on a horizontal surface"
