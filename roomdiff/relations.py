"""Spatial relations of the changes: where each changed object sits (presentation only).

SuperMap (CMU, RSS 2026) attaches On(A, B), beside and under predicates to its object
nodes and hands them to a VLM as text. RoomDiff uses the same kind of predicates only to
say where a changed object is: "on the floor", "on a surface 0.72 m above the floor",
"under a surface 0.75 m above the floor", "beside the bucket", and for a moved object
"from the surface at 0.72 m to the floor". It adds the key `relations` to every change and
`relations_info` to the document; no other field changes (types, confidences, masks and
hence the benchmark predictions stay as they are).

Heights need the levelled world frame (`recon/meta.json: levelled`: +Y up, floor at
y = 0); without it every change gets `relations: null` and `relations_info.reason` says
why. y = 0 counts as the floor only if levelling fitted a floor plane holding at least
`floor_min_inliers` of the lowest points (`floor_check`): a kitchen filmed without its
floor levels on the counter; then no support is `floor` and no height is quoted. Units are
the reconstruction's (approximately metric; Pi3X can be off by a scale).

Per change and place (removed: before; added: after; unverified: the recording that saw
it; moved: both), with the cloud of the recording in which the object is at that place
(every `cloud_stride`-th valid pixel of its keyframes, rows and columns) and the object's
own instance points (bottom / top = 1st / 99th percentile of their y; the box when there
are no points):

1. Support (`support`): cloud points in the ring around the object's footprint (its
   oriented box in x / z grown by `support_pad`, minus the footprint itself, where the
   object hides what it stands on) at heights within [bottom - support_depth, bottom +
   support_tol], minus points within `own_dist` of the instance points. The highest
   horizontal patch (`horizontal_patch`, scanning down): a height window of +-
   `support_spread` with at least `support_min_points` points covering at least
   `support_min_frac` of the ring's 2 cm cells, the best-covered window of such a run,
   whose cells hold at least `peak_frac` of their points within three times the window (a
   table top or the floor, not a slice of a wall or of the object's own unlifted side).
   Height = median y of the window, re-centred. `floor` below `floor_max` (floor known),
   else `surface`; `none` without such a patch (the object hangs, its lower part and what
   it stands on were not reconstructed, or the cloud is too sparse there).
2. Cover (`cover`): the lowest horizontal patch above the object inside its footprint
   (heights top + `cover_gap` .. top + `cover_max`, at least `cover_min_frac` of the
   footprint cells): the object stands under a table top or a shelf; "partly" when the
   patch covers less than half of the footprint.
3. Object-object relations (`objects`) among the kept changes in the same recording:
   on(A, B) if |bottom(A) - top(B)| < `on_tol` and B's footprint covers more than
   `on_overlap` of A's (SuperMap uses the IoU, which a cup on a table never reaches);
   under(A, B) if top(A) is at least `on_tol` below bottom(B) with the same overlap;
   beside(A, B) if neither, the footprints overlap by at most `on_overlap` of the smaller
   one, are less than `beside_gap` apart and the height ranges overlap. Rejected entries
   get support and cover only.
4. Moved (`moved`): the support of the before and the after place, the displacement of
   the record (`distance_m` = transform.translation_m) and its horizontal part.
5. `text`: one phrase (`relation_text`); the report and the web viewer show
   `change_text`, the same phrase naming only the changes still kept.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roomdiff.geometry import (
    clip_convex,
    convex_polygon_distance,
    obb_footprint,
    polygon_area,
    rotation_y,
    unproject_depth,
)
from roomdiff.paths import RECORDINGS, OutputPaths

CELL = 0.02  # m, horizontal cells of the coverage test
STEP = 0.005  # m, height step of the patch search
PEAK_SPAN = 3.0  # peak test: compare with the points within this many window half-widths
REFINE_ROUNDS = 3  # the window is re-centred on the median of its points this often
BOTTOM_PCT, TOP_PCT = 1.0, 99.0  # bottom / top of an object: percentiles of its points' y
PARTLY = 0.5  # a cover over less than this share of the footprint is "partly"
MAX_BESIDE = 2  # at most this many neighbours are named in the text
NO_LEVEL = ("the world frame is not levelled (recon/meta.json), so there is no floor "
            "and no height above it")


# ------------------------------------------------------------------ geometry


@dataclass
class Place:
    """One change at one place: its recording (`side`), box and height range."""

    change_id: int
    side: str  # recording in which the object is at this place: "before" | "after"
    obb: dict[str, Any]  # vertical oriented box (the AABB as one when there is no obb)
    bottom: float
    top: float
    points: np.ndarray | None  # instance points (M, 3), None if unavailable

    @property
    def footprint(self) -> np.ndarray:
        return obb_footprint(self.obb)


def box_as_obb(aabb: Sequence[Sequence[float]]) -> dict[str, Any]:
    lo, hi = np.asarray(aabb, np.float64)
    return {"center": ((lo + hi) / 2).tolist(), "extents": (hi - lo).tolist(), "yaw_deg": 0.0}


def change_places(change: Mapping[str, Any]) -> list[str]:
    """The sides of a change where the object is (moved: both)."""
    return [side for side in RECORDINGS if change.get(side)]


def make_place(change: Mapping[str, Any], side: str,
               points: Mapping[str, Mapping[int, np.ndarray]]) -> Place:
    record = change[side]
    obb = record.get("obb") or box_as_obb(record["aabb"])
    ids = record.get("instance_ids") or [record.get("instance_id")]
    own = [points.get(side, {}).get(i) for i in ids]
    own = [p for p in own if p is not None and len(p)]
    pts = np.concatenate(own).astype(np.float64) if own else None
    if pts is not None:
        bottom, top = np.percentile(pts[:, 1], [BOTTOM_PCT, TOP_PCT])
    else:
        bottom, top = record["aabb"][0][1], record["aabb"][1][1]
    return Place(int(change["id"]), side, obb, float(bottom), float(top), pts)


def box_coords(obb: Mapping[str, Any], xz: np.ndarray) -> np.ndarray:
    """World (x, z) points (n, 2) in the box's own horizontal axes (u along its x, v along
    its z axis), relative to its centre."""
    R = rotation_y(float(obb["yaw_deg"]))[np.ix_([0, 2], [0, 2])]
    return (np.asarray(xz, np.float64) - np.asarray(obb["center"], np.float64)[[0, 2]]) @ R


def horizontal_patch(
    cells: np.ndarray, y: np.ndarray, n_cells: int, lo: float, hi: float, downward: bool,
    spread: float, min_frac: float, min_points: int, peak_frac: float,
) -> dict[str, Any] | None:
    """First horizontal patch among points with cell ids `cells` and heights `y`, scanning
    window centres (every STEP) from `hi` down to `lo` (`downward`) or up. A window of
    +- `spread` qualifies with at least `min_points` points in at least `min_frac` of the
    `n_cells` cells; of each run of qualifying windows the one covering most cells (the
    mode, not the edge of a noisy surface) is re-centred on the median of its points and
    accepted if it still qualifies, stays within [lo, hi] and its cells hold at least
    `peak_frac` of their points within `PEAK_SPAN` x `spread` inside the window (a level
    surface, not a slice of a vertical one); else the scan goes on. Returns {height,
    coverage, points, spread (interquartile range of the heights)} or None."""
    if n_cells <= 0 or len(y) < min_points:
        return None
    order = np.argsort(y, kind="stable")
    y, cells = y[order], cells[order]

    def window(h: float, half: float) -> tuple[int, int]:
        return (int(np.searchsorted(y, h - half, side="left")),
                int(np.searchsorted(y, h + half, side="right")))

    def covered(h: float) -> int:
        i0, i1 = window(h, spread)
        return np.unique(cells[i0:i1]).size if i1 - i0 >= min_points else 0

    def accept(h: float) -> dict[str, Any] | None:
        i0, i1 = window(h, spread)
        for _ in range(REFINE_ROUNDS):
            h = float(np.median(y[i0:i1]))
            i0, i1 = window(h, spread)
        cells_in = np.unique(cells[i0:i1])
        if i1 - i0 < min_points or cells_in.size < min_frac * n_cells or not lo <= h <= hi:
            return None  # thinned out, or drifted out of the band (onto the object's level)
        j0, j1 = window(h, PEAK_SPAN * spread)
        if i1 - i0 < peak_frac * int(np.isin(cells[j0:j1], cells_in).sum()):
            return None
        q1, q3 = np.percentile(y[i0:i1], [25, 75])
        return {"height": h, "coverage": min(1.0, cells_in.size / n_cells),
                "points": int(i1 - i0), "spread": float(q3 - q1)}

    centres = np.arange(hi, lo - 1e-9, -STEP) if downward else np.arange(lo, hi + 1e-9, STEP)
    best: tuple[int, float] | None = None  # (cells, centre) of the current run
    for h in [*centres, None]:  # None closes the last run
        n = covered(h) if h is not None else 0
        if n and n >= min_frac * n_cells:
            best = (n, float(h)) if best is None or n > best[0] else best
            continue
        if best is not None:
            found = accept(best[1])
            if found is not None:
                return found
            best = None
    return None


def not_own(points: np.ndarray, own: np.ndarray | None, dist: float) -> np.ndarray:
    """Mask of the `points` farther than `dist` from every point of `own`."""
    if own is None or not len(own) or not len(points):
        return np.ones(len(points), bool)
    from scipy.spatial import cKDTree

    d, _ = cKDTree(own).query(points, distance_upper_bound=dist)
    return ~(d <= dist)


def _crop(cloud: np.ndarray, place: Place, pad: float, y_lo: float, y_hi: float
          ) -> tuple[np.ndarray, np.ndarray]:
    """Cloud points within the footprint grown by `pad` and heights [y_lo, y_hi]; returns
    the points and their box coordinates (u, v)."""
    corners = obb_footprint({**place.obb, "extents": [
        place.obb["extents"][0] + 2 * pad, 0.0, place.obb["extents"][2] + 2 * pad]})
    lo, hi = corners.min(0), corners.max(0)
    sel = ((cloud[:, 1] >= y_lo) & (cloud[:, 1] <= y_hi) & (cloud[:, 0] >= lo[0])
           & (cloud[:, 0] <= hi[0]) & (cloud[:, 2] >= lo[1]) & (cloud[:, 2] <= hi[1]))
    pts = cloud[sel].astype(np.float64)
    uv = box_coords(place.obb, pts[:, [0, 2]])
    w, d = place.obb["extents"][0] / 2 + pad, place.obb["extents"][2] / 2 + pad
    inside = (np.abs(uv[:, 0]) <= w) & (np.abs(uv[:, 1]) <= d)
    return pts[inside], uv[inside]


def _grid(half_w: float, half_d: float) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Cells of CELL m over [-half_w, half_w] x [-half_d, half_d]: counts and centres."""
    nu, nv = max(1, int(np.ceil(2 * half_w / CELL))), max(1, int(np.ceil(2 * half_d / CELL)))
    uc = -half_w + (np.arange(nu) + 0.5) * CELL
    vc = -half_d + (np.arange(nv) + 0.5) * CELL
    return nu, nv, uc, vc


def _cell_ids(uv: np.ndarray, half_w: float, half_d: float, nu: int, nv: int) -> np.ndarray:
    iu = np.clip(np.floor((uv[:, 0] + half_w) / CELL).astype(np.int64), 0, nu - 1)
    iv = np.clip(np.floor((uv[:, 1] + half_d) / CELL).astype(np.int64), 0, nv - 1)
    return iu * nv + iv


def find_support(cloud: np.ndarray, place: Place, params: Mapping[str, Any],
                 floor_known: bool = True) -> dict[str, Any]:
    """Support of an object at one place (module docstring, step 1): {kind: floor |
    surface | none, height, gap (bottom - height), coverage, points, spread}; never
    `floor` unless `floor_known`."""
    pad, spread = float(params["support_pad"]), float(params["support_spread"])
    lo = place.bottom - float(params["support_depth"])
    hi = place.bottom + float(params["support_tol"])
    margin = PEAK_SPAN * spread
    pts, uv = _crop(cloud, place, pad, lo - margin, hi + margin)
    w, d = place.obb["extents"][0] / 2, place.obb["extents"][2] / 2
    ring = ~((np.abs(uv[:, 0]) < w) & (np.abs(uv[:, 1]) < d))
    keep = ring & not_own(pts, place.points, float(params["own_dist"]))
    pts, uv = pts[keep], uv[keep]
    nu, nv, uc, vc = _grid(w + pad, d + pad)
    centres_u, centres_v = np.meshgrid(uc, vc, indexing="ij")
    n_ring = int((~((np.abs(centres_u) < w) & (np.abs(centres_v) < d))).sum())
    patch = horizontal_patch(_cell_ids(uv, w + pad, d + pad, nu, nv), pts[:, 1], n_ring,
                             lo, hi, True, spread, float(params["support_min_frac"]),
                             int(params["support_min_points"]), float(params["peak_frac"]))
    if patch is None:
        return {"kind": "none", "height": None, "gap": None, "coverage": None,
                "points": len(pts)}
    height = patch["height"]
    kind = "floor" if floor_known and height < float(params["floor_max"]) else "surface"
    return {"kind": kind, "height": round(height, 3), "gap": round(place.bottom - height, 3),
            "coverage": round(patch["coverage"], 3), "points": patch["points"],
            "spread": round(patch["spread"], 4)}


def find_cover(cloud: np.ndarray, place: Place, params: Mapping[str, Any]
               ) -> dict[str, Any] | None:
    """Lowest horizontal patch above an object inside its footprint (module docstring,
    step 2): {height, coverage, points, spread}, or None."""
    spread = float(params["support_spread"])
    lo = place.top + float(params["cover_gap"])
    hi = place.top + float(params["cover_max"])
    margin = PEAK_SPAN * spread
    pts, uv = _crop(cloud, place, 0.0, lo - margin, hi + margin)
    keep = not_own(pts, place.points, float(params["own_dist"]))
    pts, uv = pts[keep], uv[keep]
    w, d = place.obb["extents"][0] / 2, place.obb["extents"][2] / 2
    nu, nv, _, _ = _grid(w, d)
    patch = horizontal_patch(_cell_ids(uv, w, d, nu, nv), pts[:, 1], nu * nv, lo, hi, False,
                             spread, float(params["cover_min_frac"]),
                             int(params["support_min_points"]), float(params["peak_frac"]))
    if patch is None:
        return None
    return {"height": round(patch["height"], 3), "coverage": round(patch["coverage"], 3),
            "points": patch["points"], "spread": round(patch["spread"], 4)}


def footprint_overlap(a: Place, b: Place) -> tuple[float, float]:
    """Intersection of the two footprints over the footprint of `a`, and over the
    smaller footprint, both in [0, 1]."""
    fa, fb = a.footprint, b.footprint
    area_a, area_b = polygon_area(fa), polygon_area(fb)
    if min(area_a, area_b) <= 1e-12:
        return 0.0, 0.0
    inter = polygon_area(clip_convex(fa, fb))
    return min(1.0, inter / area_a), min(1.0, inter / min(area_a, area_b))


def object_relations(places: Sequence[Place], params: Mapping[str, Any]
                     ) -> dict[tuple[int, str], list[dict[str, Any]]]:
    """on / under / beside between the places of one recording (module docstring, step
    3), per (change id, side): [{relation, other_change_id}], on first, then under (most
    overlap first), then beside (closest first). A's footprint must be covered by B's by
    more than `on_overlap` for on(A, B) / under(A, B): a box sitting on the box of a small
    part below it (a car on its bumper) is not on it."""
    on_tol, min_overlap = float(params["on_tol"]), float(params["on_overlap"])
    beside_gap = float(params["beside_gap"])
    out: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for a in places:
        found = []
        for b in places:
            if b.change_id == a.change_id or b.side != a.side:
                continue
            of_a, of_smaller = footprint_overlap(a, b)
            if of_a > min_overlap and abs(a.bottom - b.top) < on_tol:
                found.append((0, -of_a, "on", b.change_id))
            elif of_a > min_overlap and b.bottom - a.top >= on_tol:
                found.append((1, -of_a, "under", b.change_id))
            elif of_smaller <= min_overlap and min(a.top, b.top) > max(a.bottom, b.bottom):
                gap = convex_polygon_distance(a.footprint, b.footprint)
                if gap < beside_gap:
                    found.append((2, gap, "beside", b.change_id))
        out[(a.change_id, a.side)] = [{"relation": r, "other_change_id": cid}
                                      for _, _, r, cid in sorted(found)]
    return out


# ------------------------------------------------------------------ text


def other_name(cid: int, labels: Mapping[int, str]) -> str:
    label = labels.get(cid)
    return f"the {label}" if label and label != "object" else f"change {cid}"


def _height(h: float) -> str:
    return f"{h:.2f} m"


def _on(place: Mapping[str, Any], labels: Mapping[int, str]) -> int | None:
    """Id of the kept change the object stands on, if any."""
    on = [o["other_change_id"] for o in place.get("objects") or []
          if o["relation"] == "on" and o["other_change_id"] in labels]
    return on[0] if on else None


def rest_words(place: Mapping[str, Any], labels: Mapping[int, str], floor_known: bool = True
               ) -> str | None:
    """What the object rests on: "the yellow can" (another change), "the floor", "the
    surface at 0.72 m" ("a surface" when the floor is not known); None if unknown."""
    other = _on(place, labels)
    if other is not None:
        return other_name(other, labels)
    support = place.get("support") or {}
    if support.get("kind") == "floor":
        return "the floor"
    if support.get("kind") == "surface":
        return f"the surface at {_height(support['height'])}" if floor_known else "a surface"
    return None


def place_parts(place: Mapping[str, Any], labels: Mapping[int, str],
                with_support: bool = True, floor_known: bool = True) -> list[str]:
    """Phrases of one place: support ("on ..."), then under / cover, then beside."""
    parts: list[str] = []
    objects = [o for o in place.get("objects") or [] if o["other_change_id"] in labels]
    support = place.get("support") or {}
    if with_support:
        other = _on(place, labels)
        if other is not None:
            parts.append("on " + other_name(other, labels))
        elif support.get("kind") == "floor":
            parts.append("on the floor")
        elif support.get("kind") == "surface":
            parts.append(f"on a surface {_height(support['height'])} above the floor"
                         if floor_known else "on a horizontal surface")
    under = [o for o in objects if o["relation"] == "under"]
    cover = place.get("cover")
    if under:
        parts.append("under " + other_name(under[0]["other_change_id"], labels))
    elif cover:
        partly = "partly " if cover["coverage"] < PARTLY else ""
        parts.append(f"{partly}under a surface {_height(cover['height'])} above the floor"
                     if floor_known else f"{partly}under a horizontal surface")
    beside = list(dict.fromkeys(other_name(o["other_change_id"], labels) for o in objects
                                if o["relation"] == "beside"))[:MAX_BESIDE]  # distinct names
    if beside:
        parts.append("beside " + " and ".join(beside))
    return parts


def same_support(a: Mapping[str, Any], b: Mapping[str, Any], tol: float) -> bool:
    if a.get("kind") != b.get("kind") or a.get("kind") not in ("floor", "surface"):
        return False
    return a["kind"] == "floor" or abs(a["height"] - b["height"]) < tol


def moved_text(rel: Mapping[str, Any], labels: Mapping[int, str], same_tol: float,
               floor_known: bool) -> str:
    """"from the surface at 0.72 m to the floor", "on the floor both times", "from a
    surface to one 0.30 m lower" (floor not known), ... plus the after place's under /
    beside phrases."""
    b, a = rel["before"], rel["after"]
    sb, sa = b.get("support") or {}, a.get("support") or {}
    wb, wa = rest_words(b, labels, floor_known), rest_words(a, labels, floor_known)
    if same_support(sb, sa, same_tol):  # one level: name the surface, not what is on it
        text = place_parts({"support": sa}, {}, floor_known=floor_known)[0] + " both times"
    elif wb and wa and not floor_known and sb.get("kind") == sa.get("kind") == "surface":
        drop = sa["height"] - sb["height"]  # heights are relative without a known floor
        text = f"from a surface to one {_height(abs(drop))} {'lower' if drop < 0 else 'higher'}"
    elif wb and wa:
        text = f"from {wb} to {wa}"
    else:
        text = f"from {wb}" if wb else f"to {wa}" if wa else ""
    rest = place_parts(a, labels, with_support=False, floor_known=floor_known)
    return ", ".join(t for t in (text, *rest) if t)


def relation_text(change: Mapping[str, Any], labels: Mapping[int, str],
                  same_tol: float = 0.05, floor_known: bool = True) -> str:
    """Short phrase of where the object is (moved: where it went), from its `relations`;
    other changes are named by `labels` {id: label} (the kept changes: relations to any
    other change are left out). Heights are quoted above the floor only if
    `floor_known`. "" without relations."""
    rel = change.get("relations")
    if not rel:
        return ""
    places = [s for s in RECORDINGS if rel.get(s)]
    if change.get("type") == "moved" and len(places) == 2:
        return moved_text(rel, labels, same_tol, floor_known)
    if not places:
        return ""
    return ", ".join(place_parts(rel[places[0]], labels, floor_known=floor_known))


def kept_labels(doc: Mapping[str, Any]) -> dict[int, str]:
    """{id: label} of the kept changes of a changes document."""
    return {int(c["id"]): str(c.get("label") or "object") for c in doc.get("changes", [])}


def change_text(doc: Mapping[str, Any], change: Mapping[str, Any]) -> str:
    """`relation_text` of a change of the document `doc` (its kept changes name the others,
    its `relations_info` says whether the floor is known): what the report and the viewer
    show, current even after a later stage moved a change to `rejected`."""
    info = doc.get("relations_info") or {}
    same_tol = float((info.get("params") or {}).get("same_height_tol", 0.05))
    return relation_text(change, kept_labels(doc), same_tol, bool(info.get("floor_known", True)))


# ------------------------------------------------------------------ stage


def recording_cloud(recon: Mapping[str, Any], rec_id: int, stride: int) -> np.ndarray:
    """World points (M, 3) of every `stride`-th valid pixel (rows and columns) of the
    keyframes of one recording (0 = before, 1 = after)."""
    pts = [np.zeros((0, 3), np.float32)]
    for i in np.flatnonzero(np.asarray(recon["rec"]) == rec_id):
        world = unproject_depth(recon["depth"][i], recon["K"][i], recon["cam2world"][i])
        pts.append(world[::stride, ::stride][recon["valid"][i][::stride, ::stride]])
    return np.concatenate(pts)


def instance_points(paths: OutputPaths) -> dict[str, dict[int, np.ndarray]]:
    """Instance points per recording and id from `instances/<rec>_points.npz` (empty if
    missing)."""
    out: dict[str, dict[int, np.ndarray]] = {}
    for name in RECORDINGS:
        file = paths.instances / f"{name}_points.npz"
        if file.is_file():
            with np.load(file) as f:
                pts, owner = f["points"], f["instance"]
            out[name] = {int(i): pts[owner == i] for i in np.unique(owner)}
    return out


def place_relations(cloud: np.ndarray, place: Place, params: Mapping[str, Any],
                    floor_known: bool = True) -> dict[str, Any]:
    return {"support": find_support(cloud, place, params, floor_known),
            "cover": find_cover(cloud, place, params), "objects": []}


def moved_summary(change: Mapping[str, Any], rel: Mapping[str, Any]) -> dict[str, Any]:
    def short(support: Mapping[str, Any]) -> dict[str, Any]:
        return ({"kind": support["kind"]} if support["kind"] != "surface"
                else {"kind": "surface", "height": support["height"]})

    tf = change.get("transform") or {}
    b, a = change["before"]["centroid"], change["after"]["centroid"]
    return {"from": short(rel["before"]["support"]), "to": short(rel["after"]["support"]),
            "distance_m": tf.get("translation_m"),
            "horizontal_m": round(float(np.hypot(a[0] - b[0], a[2] - b[2])), 3)}


def compute_relations(
    changes: Sequence[dict[str, Any]], clouds: Mapping[str, np.ndarray],
    points: Mapping[str, Mapping[int, np.ndarray]], params: Mapping[str, Any],
    rejected: Sequence[dict[str, Any]] = (), floor_known: bool = True,
) -> Counter:
    """Set `relations` on every change (in place): supports, covers, object relations
    among `changes` (not `rejected`), moved summary and text (`floor_known`: y = 0 is the
    floor). Returns the support counts."""
    counts: Counter = Counter()
    labels = {int(c["id"]): str(c.get("label") or "object") for c in changes}
    for group, related in ((changes, True), (rejected, False)):
        places = []
        for c in group:
            rel: dict[str, Any] = dict.fromkeys(RECORDINGS)
            for side in change_places(c):
                place = make_place(c, side, points)
                rel[side] = place_relations(clouds[side], place, params, floor_known)
                counts[rel[side]["support"]["kind"]] += 1
                places.append(place)
            c["relations"] = rel
        if related:
            found = object_relations(places, params)
            for c in group:
                for side in change_places(c):
                    c["relations"][side]["objects"] = found.get((int(c["id"]), side), [])
        for c in group:
            rel = c["relations"]
            if c.get("type") == "moved" and rel.get("before") and rel.get("after"):
                rel["moved"] = moved_summary(c, rel)
            rel["text"] = relation_text(c, labels if related else {},
                                        float(params["same_height_tol"]), floor_known)
    return counts


def floor_check(meta: Mapping[str, Any], params: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Is y = 0 of the levelled frame the floor? Only if levelling fitted a floor plane
    that holds at least `floor_min_inliers` of the floor candidates (the lowest 10 % of the
    points): when the floor was never filmed the lowest plane is a counter or a bed
    (kitchen_8: 9 %). Returns (known, note)."""
    method = (meta.get("level") or {}).get("method")
    inliers = meta.get("floor_inliers_fraction")
    if method != "ransac floor plane":
        return False, f"levelled without a floor plane ({method}): heights not quoted"
    if inliers is None or inliers < float(params["floor_min_inliers"]):
        return False, (f"the fitted floor plane holds only {100 * (inliers or 0):.0f}% of the "
                       "lowest points (maybe a counter, not the floor): heights not quoted")
    return True, None


def add_relations(
    out_dir: str | Path, doc: dict[str, Any], cfg: Mapping[str, Any],
    recon: Mapping[str, Any] | None = None,
    points: Mapping[str, Mapping[int, np.ndarray]] | None = None,
) -> dict[str, Any] | None:
    """Add `relations` to every change (and rejected entry) of the changes document `doc`
    and `relations_info` to it, in place; the reconstruction and instance points are read
    from `out_dir` unless given. Returns the info; None (doc untouched) when
    `relations.enabled` is false or the config has no `relations` section."""
    params = (cfg or {}).get("relations")
    if not params or not params.get("enabled", False):
        return None
    paths = OutputPaths(out_dir)
    entries = [*doc.get("changes", []), *doc.get("rejected", [])]
    meta_file = paths.recon / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
    reason = None
    if not meta.get("levelled", False):
        reason = NO_LEVEL
    elif recon is None and not (paths.recon / "recon.npz").is_file():
        reason = "no reconstruction (recon/recon.npz)"
    if reason is not None:
        for c in entries:
            c["relations"] = None
        info: dict[str, Any] = {"levelled": bool(meta.get("levelled", False)),
                                "reason": reason}
        doc["relations_info"] = info
        return info
    if recon is None:
        from roomdiff.recon import load_recon

        recon = load_recon(paths.recon / "recon.npz")
    stride = int(params["cloud_stride"])
    clouds = {name: recording_cloud(recon, k, stride) for k, name in enumerate(RECORDINGS)}
    if points is None:
        points = instance_points(paths)
    floor_known, note = floor_check(meta, params)
    counts = compute_relations(doc.get("changes", []), clouds, points, params,
                               doc.get("rejected", []), floor_known)
    info = {"levelled": True, "reason": None, "floor_known": floor_known, "floor_note": note,
            "floor_method": (meta.get("level") or {}).get("method"),
            "floor_inliers_fraction": meta.get("floor_inliers_fraction"),
            "supports": {k: counts[k] for k in ("floor", "surface", "none")},
            "params": dict(params)}
    doc["relations_info"] = info
    return info


def run_relations(out_dir: str | Path, cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    """`roomdiff relations`: (re)compute the relations of `<out>/changes.json` in place
    (no model); rewrite `report.md` when the document has been through M5. Returns the
    document, None when disabled."""
    paths = OutputPaths(out_dir)
    doc = json.loads(paths.changes_json.read_text())
    info = add_relations(out_dir, doc, cfg)
    if info is None:
        print("[relations] disabled (relations.enabled: false): changes.json left as it is")
        return None
    paths.changes_json.write_text(json.dumps(doc, indent=2))
    if info["reason"]:
        print(f"[relations] none: {info['reason']} -> {paths.changes_json}")
    else:
        print(f"[relations] supports {info['supports']} -> {paths.changes_json}")
    if "rejected" in doc:  # the report is written from the M5 document
        from roomdiff.report import write_report

        write_report(out_dir)
    return doc
