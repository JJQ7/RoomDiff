"""M4 changes: match before / after instances and type every change.

* Moved candidates: before-instances from disappeared blobs (D) and after-instances from
  appeared blobs (A). Cost(d, a) = (1 - cos(f_d, f_a)) + lambda_size * |log(V_d / V_a)|
  with V the volume of the oriented box (`obb`, unchanged when the world frame turns
  about +Y; the axis-aligned extents are not); Hungarian assignment over the feasible
  pairs (cos >= tau_feat and every sorted-extent ratio within [extent_ratio_min, max]).
* Matched pair -> `moved` (rigid transform: Open3D point-to-point ICP of the before points
  onto the after points, started from the centroid translation; the rotation is reported
  if the ICP fitness reaches icp_min_fitness and marked `rotation_reliable` from
  icp_reliable_fitness on), unmatched D -> `removed`, unmatched A -> `added`, unverified
  instances -> `unverified` (with the recording that saw them).
* Confidence in [0, 1] = (w_mask * S_mask + w_free * S_free + w_consistency * S_cons) /
  (w_mask + w_free + w_consistency) with S_mask = mean SAM mask score, S_free =
  min(observing_views / free_views_full, 1) of the other recording (moved: mean of both
  sides), S_cons = mean pairwise cosine of the per-view features (0.5 if only one view;
  moved: mean of both). Unverified: 0.5 * (w_mask * S_mask + w_consistency * S_cons) /
  (w_mask + w_consistency), so at most 0.5.
* Crops `<out>/crops/<id>_before.jpg` / `<id>_after.jpg` (moved also `_before_place` /
  `_after_place`): the object in its own recording with mask outline and box, and the
  other recording looking at the same place; both views are ranked by
  `roomdiff.projection.score_views` on the instance points (frame-independent).

Writes `<out>/changes.json` (docs/interfaces.md) with label "object" and a templated
description (M5 replaces both), `<out>/crops/` and `<out>/instances/matching.json`.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from roomdiff.instances import (
    decode_rle,
    enlarge_box,
    load_instances,
    load_masks,
    mask_box,
    select_views,
)
from roomdiff.paths import RECORDINGS, OutputPaths
from roomdiff.projection import score_views
from roomdiff.recon import load_recon

CHANGE_TYPES = ("moved", "removed", "added", "unverified")
TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "added": (40, 200, 60),
    "removed": (230, 40, 40),
    "moved": (50, 110, 240),
    "unverified": (235, 200, 40),
}
MIN_EXTENT = 0.01  # m, extents are clamped to this before ratios / volumes
INFEASIBLE = 1e6  # Hungarian cost of a pair that fails the thresholds
FINE_ICP_DIST = 0.03  # m, correspondence distance of the fine ICP pass and its fitness
CROP_DEPTH_TOL = 0.05  # m, depth tolerance of the crop view scoring
CROP_MIN_ASPECT = 0.6  # crops are at least this wide relative to their height (and v.v.)
UNITS = "metres (approx.)"


# ------------------------------------------------------------------ matching


def box_extents(record: Mapping[str, Any]) -> Sequence[float]:
    """Extents of an instance's oriented box (`obb`), else of its axis-aligned box."""
    return (record.get("obb") or {}).get("extents") or record["extents"]


def sorted_extents(extents: Sequence[float]) -> np.ndarray:
    return np.sort(np.maximum(np.asarray(extents, np.float64), MIN_EXTENT))


def extents_compatible(a: Sequence[float], b: Sequence[float], lo: float, hi: float) -> bool:
    """Every ratio of the sorted extents lies in [lo, hi]."""
    ratio = sorted_extents(a) / sorted_extents(b)
    return bool(np.all((ratio >= lo) & (ratio <= hi)))


def match_instances(
    feats_d: np.ndarray, feats_a: np.ndarray, ext_d: Sequence[Sequence[float]],
    ext_a: Sequence[Sequence[float]], params: Mapping[str, Any],
) -> tuple[list[tuple[int, int, float]], np.ndarray, np.ndarray]:
    """Hungarian matching of D (rows) and A (columns) instances.

    Returns accepted pairs `(i, j, cosine)`, the cosine matrix and the cost matrix
    (infeasible pairs cost `INFEASIBLE` and are never accepted).
    """
    n_d, n_a = len(feats_d), len(feats_a)
    if n_d == 0 or n_a == 0:
        return [], np.zeros((n_d, n_a)), np.zeros((n_d, n_a))
    from scipy.optimize import linear_sum_assignment

    cos = np.asarray(feats_d, np.float64) @ np.asarray(feats_a, np.float64).T
    vol_d = np.array([np.prod(sorted_extents(e)) for e in ext_d])
    vol_a = np.array([np.prod(sorted_extents(e)) for e in ext_a])
    cost = (1 - cos) + params["lambda_size"] * np.abs(np.log(vol_d[:, None] / vol_a[None]))
    for i in range(n_d):
        for j in range(n_a):
            if cos[i, j] < params["tau_feat"] or not extents_compatible(
                    ext_d[i], ext_a[j], params["extent_ratio_min"], params["extent_ratio_max"]):
                cost[i, j] = INFEASIBLE
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(i), int(j), float(cos[i, j])) for i, j in zip(rows, cols)
             if cost[i, j] < INFEASIBLE]
    return pairs, cos, cost


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def rigid_transform(
    src: np.ndarray, dst: np.ndarray, max_dist: float, min_fitness: float,
    reliable_fitness: float | None = None,
) -> dict[str, Any]:
    """Rigid motion of an object from its before points `src` to its after points `dst`.

    Open3D point-to-point ICP initialised with the centroid translation (correspondences
    up to `max_dist`), then refined with `FINE_ICP_DIST`. Fitness = the smaller of the two
    inlier fractions within `FINE_ICP_DIST` (symmetric). From `min_fitness` on the ICP
    motion is used and `rotation_deg` reported, `rotation_reliable` from
    `reliable_fitness` on (default: `min_fitness`); below `min_fitness` R = I, t =
    centroid difference and `rotation_deg` = None. `translation_m` is the displacement of
    the before centroid.
    """
    import open3d as o3d

    reg = o3d.pipelines.registration
    clouds = []
    for pts in (src, dst):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, np.float64))
        clouds.append(pcd)
    init = np.eye(4)
    init[:3, 3] = dst.mean(0) - src.mean(0)
    point_to_point = reg.TransformationEstimationPointToPoint()
    coarse = reg.registration_icp(clouds[0], clouds[1], max_dist, init, point_to_point)
    fine = reg.registration_icp(clouds[0], clouds[1], FINE_ICP_DIST, coarse.transformation,
                                point_to_point)
    T = np.asarray(fine.transformation)
    back = reg.evaluate_registration(clouds[1], clouds[0], FINE_ICP_DIST, np.linalg.inv(T))
    fitness = min(float(fine.fitness), float(back.fitness))
    ok = fitness >= min_fitness
    reliable = ok and fitness >= (min_fitness if reliable_fitness is None else reliable_fitness)
    if not ok:
        T = init
    c = src.mean(0)
    moved = T[:3, :3] @ c + T[:3, 3]
    return {
        "translation_m": round(float(np.linalg.norm(moved - c)), 4),
        "rotation_deg": round(rotation_angle_deg(T[:3, :3]), 2) if ok else None,
        "rotation_reliable": bool(reliable),
        "R": T[:3, :3].round(6).tolist(),
        "t": T[:3, 3].round(5).tolist(),
        "icp_fitness": round(fitness, 4),
        "icp_rmse": round(float(fine.inlier_rmse), 4),
    }


# ------------------------------------------------------------------ confidence


def confidence(
    mask_scores: Sequence[float], consistencies: Sequence[float | None],
    observing_views: Sequence[int] | None, params: Mapping[str, Any],
) -> float:
    """Confidence in [0, 1] (module docstring). `observing_views` = None marks an
    unverified change (result <= 0.5)."""
    s_mask = float(np.clip(np.mean(mask_scores), 0, 1)) if len(mask_scores) else 0.0
    s_cons = float(np.clip(np.mean([0.5 if c is None else c for c in consistencies]), 0, 1)) \
        if len(consistencies) else 0.5
    w_mask, w_free, w_cons = params["w_mask"], params["w_free"], params["w_consistency"]
    if observing_views is None:
        return round(0.5 * (w_mask * s_mask + w_cons * s_cons) / max(w_mask + w_cons, 1e-9), 3)
    s_free = float(np.mean([min(v / params["free_views_full"], 1.0) for v in observing_views]))
    total = w_mask * s_mask + w_free * s_free + w_cons * s_cons
    return round(total / max(w_mask + w_free + w_cons, 1e-9), 3)


# ------------------------------------------------------------------ crops


class CropMaker:
    """Draws crops from the full-resolution keyframes (recon images as a fallback)."""

    def __init__(self, paths: OutputPaths, recon: Mapping[str, Any], params: Mapping[str, Any]):
        self.paths, self.recon, self.params = paths, recon, params
        self.folder = paths.root / "crops"
        self.folder.mkdir(parents=True, exist_ok=True)
        for old in self.folder.glob("*.jpg"):
            old.unlink()
        self.masks = {}
        for name in RECORDINGS:
            path = paths.instances / f"{name}_masks.npz"
            self.masks[name] = load_masks(path) if path.exists() else {}

    def frame_mask(self, frame: int, instance_id: int) -> np.ndarray | None:
        """Stored per-keyframe SAM mask of an instance, if any."""
        rec = RECORDINGS[int(self.recon["rec"][frame])]
        for inst, _, rle in self.masks[rec].get(str(self.recon["files"][frame]), []):
            if inst == instance_id:
                return decode_rle(rle)
        return None

    def mask_score(self, frame: int, instance_id: int) -> float | None:
        """SAM score of the stored per-keyframe mask of an instance, None if there is none."""
        rec = RECORDINGS[int(self.recon["rec"][frame])]
        for inst, score, _ in self.masks[rec].get(str(self.recon["files"][frame]), []):
            if inst == instance_id:
                return score
        return None

    def object_view(self, record: Mapping[str, Any], points: np.ndarray
                    ) -> tuple[int, tuple[int, int, int, int], np.ndarray | None]:
        """Best view of an instance in its own recording: the keyframes ranked by
        `score_views` on its points (depth-consistent), those with a stored per-keyframe
        mask first (score x SAM score); the box is the mask's, else that of the projected
        points. Falls back to the prompted view with the largest mask box x score."""
        files = list(map(str, self.recon["files"]))
        frames = np.flatnonzero(self.recon["rec"] == RECORDINGS.index(record["recording"]))
        ranked = score_views(points, frames, self.recon["cam2world"], self.recon["K"],
                             self.recon["depth"], self.recon["valid"], CROP_DEPTH_TOL)
        ranked = [v for v in ranked if v.score > 0]
        if ranked:
            sam = {v.frame: self.mask_score(v.frame, record["id"]) for v in ranked}
            best = max(ranked, key=lambda v: (sam[v.frame] is not None,
                                              v.score * (sam[v.frame] or 1.0), -v.frame))
            mask = self.frame_mask(best.frame, record["id"])
            return best.frame, best.box if mask is None else mask_box(mask), mask
        view = max(record["views"], key=lambda v: (v["mask_box"][2] - v["mask_box"][0])
                   * (v["mask_box"][3] - v["mask_box"][1]) * v["score"])
        frame = files.index(view["file"])
        return frame, tuple(view["mask_box"]), self.frame_mask(frame, record["id"])

    def place_view(
        self, points: np.ndarray, rec: int
    ) -> tuple[int, tuple[int, int, int, int]] | None:
        """Keyframe of recording `rec` that sees the place of `points` best (not occluded)."""
        frames = np.flatnonzero(self.recon["rec"] == rec)
        chosen = select_views(points, frames, self.recon, 1, CROP_DEPTH_TOL, 0.0,
                              on_surface=False)
        return chosen[0] if chosen else None

    def save(self, name: str, frame: int, box: Sequence[int], mask: np.ndarray | None,
             color: tuple[int, int, int]) -> str:
        """Write `crops/<name>.jpg`; returns its path relative to the output directory."""
        from PIL import Image, ImageDraw

        rec = RECORDINGS[int(self.recon["rec"][frame])]
        small = self.recon["images"][frame]
        full = self.paths.keyframes_of(rec) / str(self.recon["files"][frame])
        image = Image.open(full).convert("RGB") if full.exists() else Image.fromarray(small)
        scale = image.width / small.shape[1]
        if mask is not None:
            big = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
                image.size, Image.Resampling.NEAREST)) > 0
            edge = big & ~ndimage.binary_erosion(big, iterations=max(2, round(2 * scale)))
            pixels = np.asarray(image).copy()
            pixels[edge] = color
            image = Image.fromarray(pixels)
        big_box = [v * scale for v in box]
        draw = ImageDraw.Draw(image)
        draw.rectangle(big_box, outline=color, width=max(2, round(2 * scale)))
        side = max(big_box[2] - big_box[0], big_box[3] - big_box[1]) * self.params["crop_context"]
        x0, y0, x1, y1 = enlarge_box(big_box, self.params["crop_context"] - 1, image.width,
                                     image.height,
                                     max(self.params["crop_min_px"], CROP_MIN_ASPECT * side))
        image.crop((x0, y0, x1 + 1, y1 + 1)).save(self.folder / f"{name}.jpg", quality=90)
        return f"crops/{name}.jpg"

    def object_crop(self, name: str, record: Mapping[str, Any], points: np.ndarray,
                    color: tuple[int, int, int]) -> str:
        frame, box, mask = self.object_view(record, points)
        return self.save(name, frame, box, mask, color)

    def place_crop(self, name: str, points: np.ndarray, rec: int,
                   color: tuple[int, int, int]) -> str | None:
        view = self.place_view(points, rec)
        return None if view is None else self.save(name, view[0], view[1], None, color)


# ------------------------------------------------------------------ records


def side_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """`before` / `after` entry of a change from an instance record."""
    return {"instance_id": record["id"], "centroid": record["centroid"], "aabb": record["aabb"],
            "extents": record["extents"], "obb": record.get("obb"),
            "n_points": record["n_points"], "blob_ids": record["blob_ids"]}


def _xyz(v: Sequence[float]) -> str:
    return "(" + ", ".join(f"{x:.2f}" for x in v) + ")"


def _size(extents: Sequence[float]) -> str:
    return " x ".join(f"{e:.2f}" for e in extents)


def describe(kind: str, change: Mapping[str, Any]) -> str:
    """Templated one-sentence description (M5 replaces it with the VLM's)."""
    ev = change["evidence"]
    if kind == "removed":
        b = change["before"]
        return (f"An object about {_size(b['extents'])} m was removed from {_xyz(b['centroid'])}; "
                f"the after recording saw that space empty in {ev['observing_views']} views.")
    if kind == "added":
        a = change["after"]
        return (f"An object about {_size(a['extents'])} m was added at {_xyz(a['centroid'])}; "
                f"the before recording saw that space empty in {ev['observing_views']} views.")
    if kind == "moved":
        b, a, tf = change["before"], change["after"], change["transform"]
        rot = (" (rotation not determined)" if tf["rotation_deg"] is None else
               f" and turned by about {tf['rotation_deg']:.0f} degrees"
               if tf.get("rotation_reliable", True) else " (rotation uncertain)")
        return (f"An object about {_size(b['extents'])} m moved {tf['translation_m']:.2f} m from "
                f"{_xyz(b['centroid'])} to {_xyz(a['centroid'])}{rot}.")
    side = change[ev["seen_in"]]
    other = "after" if ev["seen_in"] == "before" else "before"
    return (f"An object about {_size(side['extents'])} m at {_xyz(side['centroid'])} was seen "
            f"only in the {ev['seen_in']} recording; the {other} recording never observed that "
            "space, so no verdict is possible.")


def summarize(changes: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(c["type"] for c in changes)
    verified = sum(counts[t] for t in ("moved", "added", "removed"))
    text = (f"{verified} changes detected: {counts['moved']} moved, {counts['added']} added, "
            f"{counts['removed']} removed")
    n = counts["unverified"]
    return text + f"; {n} region{'s' if n != 1 else ''} unverified."


def scene_info(paths: OutputPaths) -> dict[str, str]:
    """Video file names (from keyframes/index.json) and units."""
    info = {}
    if paths.keyframes_index.is_file():
        index = json.loads(paths.keyframes_index.read_text())
        for rec in RECORDINGS:
            video = index.get(rec, {}).get("video", {})
            if "path" in video:
                info[rec] = Path(video["path"]).name
    return {**info, "units": UNITS}


def instance_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Evidence of a removed / added / unverified change from its instance record."""
    ev = record["evidence"]
    return {"free_rays": ev["free_rays"], "free_frames_mean": ev["free_frames_mean"],
            "observing_views": ev["observing_views"], "occ_points": ev["occ_points"],
            "mask_score": record["mask_score_mean"],
            "view_consistency": record["view_consistency"]}


# ------------------------------------------------------------------ stage


def run_changes(out_dir: str | Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Match the instances of `<out>/instances/`, type the changes and write
    `<out>/changes.json` and `<out>/crops/`. Returns the changes document."""
    params = dict(cfg["changes"])
    paths = OutputPaths(out_dir)
    start = time.perf_counter()
    recon = load_recon(paths.recon / "recon.npz")
    data = {name: load_instances(out_dir, name) for name in RECORDINGS}
    bef, aft = data["before"], data["after"]
    d_idx = [k for k, r in enumerate(bef["instances"]) if r["kind"] == "disappeared"]
    a_idx = [k for k, r in enumerate(aft["instances"]) if r["kind"] == "appeared"]
    pairs, cos, cost = match_instances(
        bef["feats"][d_idx], aft["feats"][a_idx],
        [box_extents(bef["instances"][k]) for k in d_idx],
        [box_extents(aft["instances"][k]) for k in a_idx], params)

    entries: list[tuple[str, dict[str, Any]]] = []  # (type, change without id / crops)
    matched_d, matched_a = set(), set()
    for i, j, c in pairs:
        rb, ra = bef["instances"][d_idx[i]], aft["instances"][a_idx[j]]
        matched_d.add(d_idx[i])
        matched_a.add(a_idx[j])
        transform = rigid_transform(bef["points"][rb["id"]], aft["points"][ra["id"]],
                                    params["icp_max_dist"], params["icp_min_fitness"],
                                    params["icp_reliable_fitness"])
        evidence = {
            "free_rays": rb["evidence"]["free_rays"] + ra["evidence"]["free_rays"],
            "observing_views": round((rb["evidence"]["observing_views"]
                                          + ra["evidence"]["observing_views"]) / 2),
            "observing_views_before_place": rb["evidence"]["observing_views"],
            "observing_views_after_place": ra["evidence"]["observing_views"],
            "appearance_similarity": round(c, 4),
            "cls_similarity": round(float(bef["cls"][d_idx[i]] @ aft["cls"][a_idx[j]]), 4),
            "mask_score": round((rb["mask_score_mean"] + ra["mask_score_mean"]) / 2, 4),
            "icp_fitness": transform["icp_fitness"],
        }
        conf = confidence([rb["mask_score_mean"], ra["mask_score_mean"]],
                          [rb["view_consistency"], ra["view_consistency"]],
                          [rb["evidence"]["observing_views"], ra["evidence"]["observing_views"]],
                          params)
        entries.append(("moved", {"before": side_record(rb), "after": side_record(ra),
                                  "transform": transform, "evidence": evidence,
                                  "confidence": conf}))
    for doc, kind, side, matched in ((bef, "removed", "before", matched_d),
                                     (aft, "added", "after", matched_a)):
        for k, r in enumerate(doc["instances"]):
            if k in matched or r["kind"] == "unverified":
                continue
            conf = confidence([r["mask_score_mean"]], [r["view_consistency"]],
                              [r["evidence"]["observing_views"]], params)
            entries.append((kind, {side: side_record(r), "evidence": instance_evidence(r),
                                   "confidence": conf}))
    for name, doc in (("before", bef), ("after", aft)):
        for r in doc["instances"]:
            if r["kind"] != "unverified":
                continue
            conf = confidence([r["mask_score_mean"]], [r["view_consistency"]], None, params)
            entries.append(("unverified", {name: side_record(r),
                                           "evidence": {**instance_evidence(r), "seen_in": name},
                                           "confidence": conf}))
    order = {t: k for k, t in enumerate(CHANGE_TYPES)}
    entries.sort(key=lambda e: (order[e[0]], -e[1]["confidence"]))

    crops = CropMaker(paths, recon, params)
    changes = []
    for cid, (kind, body) in enumerate(entries, start=1):
        change = {"id": cid, "type": kind, "label": "object", "confidence": body["confidence"],
                  "before": body.get("before"), "after": body.get("after"),
                  "transform": body.get("transform"),
                  "evidence": {**body["evidence"], "vlm_verdict": None}}
        change["description"] = describe(kind, change)
        change["crops"] = make_crops(crops, cid, kind, change, data)
        changes.append(change)

    doc = {"scene": scene_info(paths), "changes": changes, "summary": summarize(changes)}
    paths.changes_json.write_text(json.dumps(doc, indent=2))
    matching = {
        "before_candidates": [bef["instances"][k]["id"] for k in d_idx],
        "after_candidates": [aft["instances"][k]["id"] for k in a_idx],
        "cosine": np.round(cos, 4).tolist(),
        "cost": np.round(np.minimum(cost, INFEASIBLE), 4).tolist(),
        "accepted": [[bef["instances"][d_idx[i]]["id"], aft["instances"][a_idx[j]]["id"],
                      round(c, 4)] for i, j, c in pairs],
        "params": params,
        "runtime_s": round(time.perf_counter() - start, 2),
    }
    (paths.instances / "matching.json").write_text(json.dumps(matching, indent=1))
    print(f"[changes] {doc['summary']} ({matching['runtime_s']:.1f} s) -> {paths.changes_json}")
    return doc


def make_crops(
    crops: CropMaker, cid: int, kind: str, change: Mapping[str, Any],
    data: Mapping[str, Mapping[str, Any]],
) -> dict[str, str | None]:
    """Crop files of one change (paths relative to the output directory)."""
    color = TYPE_COLORS[kind]
    out: dict[str, str | None] = {}
    for rec, name in enumerate(RECORDINGS):
        side = change[name]
        other = RECORDINGS[1 - rec]
        if side is None:
            continue
        record = next(r for r in data[name]["instances"] if r["id"] == side["instance_id"])
        points = data[name]["points"][side["instance_id"]]
        out[name] = crops.object_crop(f"{cid}_{name}", record, points, color)
        if kind == "moved":  # the other recording at this place (empty there)
            out[f"{name}_place"] = crops.place_crop(f"{cid}_{name}_place", points, 1 - rec, color)
        elif change[other] is None:
            out[other] = crops.place_crop(f"{cid}_{other}", points, 1 - rec, color)
    return out
