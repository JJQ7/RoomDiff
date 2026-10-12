"""Tests for M4 instances (CPU, fake models, synthetic geometry)."""

import numpy as np
import pytest

from roomdiff.config import load_config
from roomdiff.instances import (
    Instance,
    View,
    box_distance,
    build_instance,
    decode_rle,
    encode_rle,
    enlarge_box,
    keep_near_blob,
    mean_pairwise_cosine,
    merge_close_fragments,
    merge_instances,
    pick_mask,
    pooled_feature,
    prompt_points,
    select_blobs,
)
from roomdiff.models import FakeFeaturizer, FakeSegmenter

H, W = 320, 320
K = np.array([[200.0, 0.0, 159.5], [0.0, 200.0, 159.5], [0.0, 0.0, 1.0]], np.float32)
FLOOR_Y = 0.5  # OpenCV world: y down, the floor 0.5 m below the cameras
BOX_LO, BOX_HI = np.array([-0.15, 0.25, 0.85]), np.array([0.15, FLOOR_Y, 1.15])  # on the floor
PARAMS = dict(load_config().instances)


def pose(x: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = x
    return T


def render(cam2world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z-depth and object id (0 floor / none, 1 box) of the floor + box scene."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1)
    c = cam2world[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(d[..., 1] > 0, (FLOOR_Y - c[1]) / d[..., 1], np.inf)
        t0, t1 = (BOX_LO - c) / d, (BOX_HI - c) / d
    near, far = np.minimum(t0, t1).max(-1), np.maximum(t0, t1).min(-1)
    hit_box = (near < far) & (near > 0) & (near < t_floor)
    depth = np.where(hit_box, near, t_floor)
    depth = np.where(np.isfinite(depth) & (depth < 6), depth, 0.0)
    return depth.astype(np.float32), hit_box.astype(np.int32)


def make_recon(xs=(-0.4, 0.0, 0.4)) -> tuple[dict, dict[str, np.ndarray]]:
    """Recon-like dict of cameras at the lateral positions `xs`, plus label maps."""
    data = {k: [] for k in ("rec", "files", "images", "cam2world", "K", "depth", "valid")}
    labels = {}
    for i, x in enumerate(xs):
        T = pose(x)
        depth, ids = render(T)
        image = np.where(ids[..., None] == 1, [200, 30, 30], [90, 90, 90]).astype(np.uint8)
        data["rec"].append(0)
        data["files"].append(f"before_{i:05d}.jpg")
        data["images"].append(image)
        data["cam2world"].append(T)
        data["K"].append(K)
        data["depth"].append(depth)
        data["valid"].append(depth > 0)
        labels[str(i)] = ids
    return {k: np.stack(v) for k, v in data.items()}, labels


def box_surface_voxels(step: float = 0.02) -> np.ndarray:
    """Voxel centres on the visible faces of the box (front face and top), as M3 gives."""
    xs = np.arange(BOX_LO[0] + step / 2, BOX_HI[0], step)
    ys = np.arange(BOX_LO[1] + step / 2, BOX_HI[1] - 0.04, step)  # not touching the floor
    zs = np.arange(BOX_LO[2] + step / 2, BOX_HI[2], step)
    front = np.array([[x, y, BOX_LO[2] + step / 2] for x in xs for y in ys])
    top = np.array([[x, BOX_LO[1] + step / 2, z] for x in xs for z in zs])
    return np.concatenate([front, top])


def on_floor(points: np.ndarray) -> np.ndarray:
    """Static-surface test of the synthetic scene: the floor is in both recordings."""
    return np.abs(points[:, 1] - FLOOR_Y) < 0.03


def inside_box(points: np.ndarray, tol: float) -> np.ndarray:
    return np.all((points >= BOX_LO - tol) & (points <= BOX_HI + tol), axis=1)


# ------------------------------------------------------------------ blob choice


def blob(id_, kind, centroid, half=0.1, volume=0.01):
    c = np.asarray(centroid, float)
    return {"id": id_, "type": kind, "volume_m3": volume, "voxels": 100, "centroid": c.tolist(),
            "aabb": [(c - half).tolist(), (c + half).tolist()],
            "evidence": {"free_frames_mean": 3.0, "occ_points": 50, "observing_views": 4}}


def test_hidden_surface_rule_and_size_filters():
    candidates = [
        blob(1, "disappeared", [0, 0, 2]),
        blob(2, "unverified_after", [0, 0, 2.25]),  # 0.15 m behind the removed box
        blob(3, "unverified_after", [1.0, 0, 2]),  # far from it
        blob(4, "unverified_before", [-1.0, 0, 2], volume=0.001),  # too small
        blob(5, "unverified_before", [0, -1.0, 2], half=1.0),  # 2 m region, not an object
        blob(6, "appeared", [3, 0, 0], volume=0.0001),  # verified blobs are always kept
    ]
    kept, dropped = select_blobs(candidates, 0.004, 0.20, 1.5)
    assert [b["id"] for b in kept] == [1, 3, 6]
    assert {d["blob"]: d["reason"] for d in dropped} == {
        2: "hidden surface", 4: "small unverified", 5: "unverified region too large"}
    kept, _ = select_blobs(candidates, 0.004, 0.10, 1.5)  # smaller margin keeps blob 2
    assert 2 in [b["id"] for b in kept]
    assert box_distance([0, 0, 0], [-1, -1, -1], [1, 1, 1]) == 0.0
    assert box_distance([3, 0, 0], [-1, -1, -1], [1, 1, 1]) == pytest.approx(2.0)


# ------------------------------------------------------------------ lifting


class Bleeding(FakeSegmenter):
    """Returns the prompt box grown by 8 px sideways and 30 px downwards: a mask that
    bleeds onto the floor (still mostly inside the prompt box)."""

    def segment(self, key, image, box, points):
        x0, y0, x1, y1 = box
        return super().segment(key, image, (x0 - 8, y0, x1 + 8, y1 + 30), None)


def test_lift_keeps_the_object_and_drops_mask_bleeding():
    from scipy.spatial import cKDTree

    recon, _ = make_recon()
    voxels = box_surface_voxels()
    seg = Bleeding()
    inst, reason = build_instance(blob(7, "disappeared", voxels.mean(0)), voxels, recon,
                                  [0, 1, 2], seg, PARAMS, on_floor)
    assert reason == "ok" and inst is not None and inst.kind == "disappeared"
    assert seg.calls == len(inst.views) == 3  # n_views = 4 but only 3 cameras
    assert len(inst.points) >= PARAMS["min_points"]
    # floor points survive only within near_voxel_m of the blob voxels
    dist = cKDTree(voxels).query(inst.points)[0]
    assert np.all(inside_box(inst.points, 0.03) | (dist <= PARAMS["near_voxel_m"] + 0.01))
    assert inside_box(inst.points, 0.03).mean() > 0.8
    # without the static-surface test the whole expanded box of floor comes along
    loose, _ = build_instance(blob(7, "disappeared", voxels.mean(0)), voxels, recon,
                              [0, 1, 2], Bleeding(), PARAMS, None)
    assert loose is not None and len(loose.points) > len(inst.points)
    assert cKDTree(voxels).query(loose.points)[0].max() > 0.13
    assert inside_box(loose.points, 0.03).mean() < inside_box(inst.points, 0.03).mean() - 0.1


def test_keep_near_blob_rules():
    voxels = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    points = np.array([
        [0.05, 0.0, 0.0],  # between the voxels
        [0.25, 0.0, 0.0],  # in the expanded box, not static
        [0.25, 0.5, 0.0],  # outside the expanded box
        [0.0, 0.0, 0.14],  # static but in the box: dropped unless near a voxel ...
        [0.0, 0.0, 0.08],  # ... like this one
    ])

    def static(p):
        return p[:, 2] > 0.05

    keep = keep_near_blob(points, voxels, expand_m=0.15, near_voxel_m=0.10, on_static=static)
    assert keep.tolist() == [True, True, False, False, True]
    assert keep_near_blob(points[:0], voxels, 0.15, 0.1).shape == (0,)


def test_select_views_skips_cameras_that_cannot_see_the_blob():
    recon, _ = make_recon(xs=(-0.4, 0.0, 0.4, 40.0))  # the last camera looks elsewhere
    voxels = box_surface_voxels()
    inst, _ = build_instance(blob(7, "disappeared", voxels.mean(0)), voxels, recon,
                             [0, 1, 2, 3], FakeSegmenter(), PARAMS, on_floor)
    assert inst is not None and sorted(v.frame for v in inst.views) == [0, 1, 2]


def test_no_mask_inside_box_drops_the_blob():
    recon, labels = make_recon()
    voxels = box_surface_voxels()
    floor_only = {k: np.where(v == 1, 0, 1) for k, v in labels.items()}  # wrong object

    class Everything(FakeSegmenter):
        def segment(self, key, image, box, points):
            return np.ones((1, *image.shape[:2]), bool), np.array([0.9])

    inst, reason = build_instance(blob(7, "disappeared", voxels.mean(0)), voxels, recon,
                                  [0, 1, 2], Everything(floor_only), PARAMS, on_floor)
    assert inst is None and reason == "no SAM mask inside the box"


# ------------------------------------------------------------------ merging


def fragment(points: np.ndarray, frame: int = 0, mask: np.ndarray | None = None,
             kind: str = "disappeared") -> Instance:
    mask = np.zeros((H, W), bool) if mask is None else mask
    view = View(frame, (0, 0, 10, 10), mask, 0.9)
    b = blob(len(points), kind, points.mean(0))
    return Instance(0, kind, [b], points, points.astype(np.float32),
                    np.zeros((len(points), 3), np.uint8), [view])


def grid_points(lo, hi, step=0.01):
    axes = [np.arange(a, b + 1e-9, step) for a, b in zip(lo, hi)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def test_merge_reassembles_fragments_and_keeps_neighbours_apart():
    seat = grid_points([0, 0.40, 0], [0.4, 0.42, 0.4])
    back = grid_points([0, 0.0, 0.38], [0.4, 0.40, 0.40])  # touches the seat at one edge
    legs = grid_points([0, 0.42, 0], [0.02, 0.8, 0.02])
    seat_plus = np.concatenate([seat, back[back[:, 1] > 0.2]])  # half of the back fragment
    neighbour = grid_points([0.55, 0.40, 0], [0.9, 0.42, 0.4])  # 15 cm away
    frags = [fragment(seat_plus), fragment(back), fragment(neighbour, kind="unverified")]
    merged = merge_instances(frags, 0.5, 0.03, 0.3, 0.01)
    assert len(merged) == 2
    chair = max(merged, key=lambda i: len(i.points))
    assert chair.kind == "disappeared" and len(chair.blobs) == 2
    assert chair.points[:, 1].min() < 0.01 and chair.points[:, 1].max() > 0.41
    # legs share no points with the seat, but their masks coincide in a shared keyframe
    mask = np.zeros((H, W), bool)
    mask[50:150, 60:160] = True
    merged = merge_instances([fragment(seat, 3, mask), fragment(legs, 3, mask)], 0.5, 0.03,
                             0.3, 0.01)
    assert len(merged) == 1 and len(merged[0].views) == 1
    other = mask.copy()
    other[50:150, 60:140] = False
    merged = merge_instances([fragment(seat, 3, mask), fragment(legs, 3, other)], 0.5, 0.03,
                             0.3, 0.01)
    assert len(merged) == 2


def test_gap_merge_joins_touching_parts_of_one_object():
    """A chair seat and its base (column + foot plate) 2.5 cm below it: too little overlap
    for the mask / point rule, but 6 % of the seat points lie within 5 cm of the base."""
    seat = grid_points([0, 0.40, 0], [0.4, 0.42, 0.4])
    base = np.concatenate([grid_points([0.18, 0.445, 0.18], [0.22, 0.8, 0.22]),
                           grid_points([0, 0.78, 0], [0.4, 0.8, 0.4])])
    neighbour = grid_points([0.5, 0.40, 0], [0.9, 0.42, 0.4])  # 10 cm beside the seat
    frags = merge_instances([fragment(seat), fragment(base), fragment(neighbour)], 0.5, 0.03,
                            0.3, 0.01)
    assert len(frags) == 3  # the overlap rule alone keeps the parts apart
    merged, records = merge_close_fragments(frags, 0.05, 0.05, 1.5, 0.01)
    assert len(merged) == 2
    chair = max(merged, key=lambda i: len(i.points))
    assert chair.kind == "disappeared" and len(chair.blobs) == 2
    assert sorted(chair.merged_from) == sorted([[len(seat)], [len(base)]])
    assert chair.points[:, 1].min() < 0.401 and chair.points[:, 1].max() > 0.79
    assert len(records) == 1 and records[0]["recording"] == "before"
    assert records[0]["near_fraction"] >= 0.05 and max(records[0]["union_obb_extents"]) < 0.5
    other = next(i for i in merged if i is not chair)
    assert other.merged_from == [] and len(other.points) == len(neighbour)
    # different kinds (an unverified base) are never joined
    merged, records = merge_close_fragments(
        [fragment(seat), fragment(base, kind="unverified")], 0.05, 0.05, 1.5, 0.01)
    assert len(merged) == 2 and records == []
    # a union longer than merge_max_extent_m: two planks end to end, 3 cm apart
    plank = grid_points([0, 0.40, 0], [1.4, 0.42, 0.04])
    short = grid_points([1.43, 0.40, 0], [1.63, 0.42, 0.04])
    assert len(merge_close_fragments([fragment(plank), fragment(short)], 0.05, 0.05, 1.5,
                                     0.01)[0]) == 2
    assert len(merge_close_fragments([fragment(plank), fragment(short)], 0.05, 0.05, 2.0,
                                     0.01)[0]) == 1


# ------------------------------------------------------------------ features, masks


def test_pooled_feature_uses_mask_patches():
    feat = FakeFeaturizer()
    image = np.full((H, W, 3), 90, np.uint8)
    image[80:160, 100:200] = (200, 30, 30)
    mask = np.zeros((H, W), bool)
    mask[80:160, 100:200] = True
    red = pooled_feature(feat.patch_tokens("a", image), mask, 3)
    assert red @ feat.cls_token(image[80:160, 100:200]) > 0.999
    tiny = np.zeros((H, W), bool)
    tiny[100, 150] = True  # no patch centre inside: falls back to the centroid patch
    assert np.linalg.norm(pooled_feature(feat.patch_tokens("a", image), tiny, 3)) == \
        pytest.approx(1.0)
    assert mean_pairwise_cosine([red]) is None
    assert mean_pairwise_cosine([red, red, red]) == pytest.approx(1.0)


def test_rle_roundtrip_and_mask_choice():
    mask = np.zeros((H, W), bool)
    mask[10:50, 20:90] = True
    rle = encode_rle(mask)
    assert rle["size"] == [H, W] and isinstance(rle["counts"], str)
    assert np.array_equal(decode_rle(rle), mask)
    outside = np.zeros((H, W), bool)
    outside[100:200, 200:300] = True
    picked = pick_mask(np.stack([outside, mask]), np.array([0.99, 0.8]), (15, 5, 95, 55))
    assert picked is not None and picked[1] == pytest.approx(0.8)  # the best is outside
    assert pick_mask(np.stack([outside]), np.array([0.99]), (15, 5, 95, 55)) is None


def test_box_and_prompt_helpers():
    assert enlarge_box((100, 100, 200, 150), 0.1, W, H) == (95, 98, 205, 152)
    x0, y0, x1, y1 = enlarge_box((300, 220, 319, 239), 0.0, W, H, min_size=100)
    assert x1 <= W - 1 and y1 <= H - 1 and x1 - x0 == 100 and y1 - y0 == 100
    cols, rows = np.mgrid[0:50, 0:30]
    pts = prompt_points(cols.ravel() + 100, rows.ravel() + 50, 5)
    assert len(pts) == 5 and np.all((pts[:, 0] >= 100) & (pts[:, 0] < 150))
    assert len(prompt_points(np.array([1, 2]), np.array([3, 4]), 5)) == 2
