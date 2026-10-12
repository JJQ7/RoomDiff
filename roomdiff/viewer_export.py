"""M6 web viewer export: a self-contained static three.js viewer in `<out>/viewer/`.

Writes
* `before.ply`, `after.ply`: the valid points of `recon.npz` with keyframe colours, voxel-
  downsampled with one shared voxel size, grown until each cloud has <= `target_points`;
* `changes.json`: `<out>/changes.json` if it exists, else changes generated from the M3
  `visibility/candidates.json`, else an empty change list (schema: docs/interfaces.md);
* `crops/`: the crop images referenced by `changes.json`;
* `scene.json`: manifest read by the viewer (file names, point counts, extent, up axis);
* a copy of the viewer app (repo `viewer/`: index.html, app.js, style.css).

Browsers refuse `fetch` on file:// pages, so serve the folder:
`cd <out>/viewer && python -m http.server 8000`, then open http://localhost:8000/.

Run: `python -m roomdiff.viewer_export <out_dir> [--target-points N]`
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from roomdiff.geometry import voxel_downsample
from roomdiff.paths import RECORDINGS, OutputPaths
from roomdiff.recon import load_recon, recording_points, write_ply

VIEWER_SRC = Path(__file__).resolve().parent.parent / "viewer"
APP_FILES = ("index.html", "app.js", "style.css")
TARGET_POINTS = 400_000
START_VOXEL = 0.005  # m, first voxel size tried by `downsample_to_target`
MIN_YAW_SPREAD = 0.1  # min singular value ratio of the camera x axes to trust 2 directions
UNITS = "metres (approx.)"
# M3 candidate type -> (change type, recording in which the region is occupied)
CANDIDATE_TO_CHANGE = {
    "disappeared": ("removed", "before"),
    "appeared": ("added", "after"),
    "unverified_before": ("unverified", "before"),
    "unverified_after": ("unverified", "after"),
}
CANDIDATE_TEXT = {
    "disappeared": "occupied before, seen empty after",
    "appeared": "occupied after, seen empty before",
    "unverified_before": "occupied before, never observed after",
    "unverified_after": "occupied after, never observed before",
}

Cloud = tuple[np.ndarray, np.ndarray]  # points (M,3) float32, colours (M,3) uint8


# ------------------------------------------------------------------ point clouds


def downsample_to_target(
    clouds: list[Cloud], target_points: int, voxel: float = START_VOXEL
) -> tuple[list[Cloud], float]:
    """Voxel-downsample all `clouds` with one shared voxel size, enlarged until every
    cloud has at most `target_points` points. Returns the clouds and the voxel size.

    The grid is anchored at the common minimum corner, so once the voxel exceeds the
    extent every cloud is a single point and the loop always ends.
    """
    if target_points < 1:
        raise ValueError(f"target_points must be positive, got {target_points}")
    nonempty = [p for p, _ in clouds if len(p)]
    origin = np.min([p.min(0) for p in nonempty], 0) if nonempty else np.zeros(3, np.float32)
    while True:
        out = []
        for points, colors in clouds:
            small, small_colors = voxel_downsample(points - origin, colors, voxel)
            out.append((small + origin, small_colors))
        largest = max(len(p) for p, _ in out)
        if largest <= target_points:
            return out, voxel
        # points on surfaces: count ~ 1 / voxel^2
        voxel *= max(1.1, 1.02 * math.sqrt(largest / target_points))


def estimate_up(cam2world: np.ndarray) -> np.ndarray:
    """Unit vector against gravity, from the camera axes (OpenCV: y down, x right).

    Gravity is the mean camera "down" (y) axis, but phones are usually tilted down while
    filming, which biases that mean (by ~7 degrees on scenediff_living_room_1). An upright
    phone keeps its x axis horizontal, so the components of the mean y axis along the
    directions spanned by the camera x axes are removed first (the second direction only
    if the camera also turned sideways). Same idea as `scripts/plot_align.py`.
    """
    _, s, vt = np.linalg.svd(cam2world[:, :3, 0].astype(np.float64))
    horizontal = vt[:2] if s[1] > MIN_YAW_SPREAD * s[0] else vt[:1]
    down = cam2world[:, :3, 1].astype(np.float64).mean(0)
    down -= horizontal.T @ (horizontal @ down)
    norm = np.linalg.norm(down)
    return -down / norm if norm > 1e-6 else np.array([0.0, -1.0, 0.0])


def horizontal_view_dir(cam2world: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Mean camera viewing direction (z axis) projected onto the floor plane, unit length.

    The viewer looks at the scene from behind the cameras along this direction."""
    forward = cam2world[:, :3, 2].astype(np.float64).mean(0)
    forward -= forward.dot(up) * up
    if np.linalg.norm(forward) < 1e-6:  # cameras look around evenly: any horizontal axis
        forward = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 0.0, 1.0])
    return forward / np.linalg.norm(forward)


# ---------------------------------------------------------------------- changes


def video_names(paths: OutputPaths) -> dict[str, str]:
    """Video file name per recording from `keyframes/index.json` (empty if missing)."""
    if not paths.keyframes_index.is_file():
        return {}
    index = json.loads(paths.keyframes_index.read_text())
    return {r: Path(index[r]["video"]["path"]).name for r in RECORDINGS
            if "path" in index.get(r, {}).get("video", {})}


def describe_candidate(blob: dict[str, Any]) -> str:
    """One-line description of an M3 candidate blob from its geometry and evidence."""
    lo, hi = np.asarray(blob["aabb"], np.float64)
    size = " x ".join(f"{s:.2f}" for s in hi - lo)
    text = f"Region {CANDIDATE_TEXT[blob['type']]}, box {size} m"
    evidence = blob.get("evidence", {})
    if "observing_views" in evidence:
        text += f", seen through by {evidence['observing_views']} views of the other video"
    return text + "."


def changes_from_candidates(candidates: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
    """changes.json (docs/interfaces.md) from M3 `candidates.json`: every blob becomes one
    change labelled "region" whose box sits in the recording where it is occupied; the
    first debug view of each recording (if any) serves as its crop."""
    changes = []
    for blob in candidates["candidates"]:
        if blob["type"] not in CANDIDATE_TO_CHANGE:
            print(f"[viewer] skipping candidate {blob.get('id')} of unknown type {blob['type']}")
            continue
        change_type, side = CANDIDATE_TO_CHANGE[blob["type"]]
        crops = {rec: f"visibility/{views[0]['crop']}"
                 for rec, views in blob.get("views", {}).items() if views}
        changes.append({
            "id": blob["id"],
            "type": change_type,
            "label": "region",
            "confidence": blob.get("confidence"),
            side: {"centroid": blob["centroid"], "aabb": blob["aabb"]},
            "evidence": blob.get("evidence", {}),
            "description": describe_candidate(blob),
            "crops": crops,
        })
    counts = Counter(c["type"] for c in changes)
    listed = ", ".join(f"{counts[t]} {t}" for t in ("removed", "added", "unverified") if counts[t])
    summary = (f"{len(changes)} candidate regions from the visibility stage"
               f"{': ' + listed if listed else ''} (geometry only, not yet verified per object).")
    return {"scene": scene, "changes": changes, "summary": summary}


def load_changes(paths: OutputPaths) -> dict[str, Any]:
    """`<out>/changes.json`, else changes generated from `visibility/candidates.json`,
    else an empty change list."""
    if paths.changes_json.is_file():
        return json.loads(paths.changes_json.read_text())
    scene = {**video_names(paths), "units": UNITS}
    candidates = paths.visibility / "candidates.json"
    if candidates.is_file():
        return changes_from_candidates(json.loads(candidates.read_text()), scene)
    return {"scene": scene, "changes": [],
            "summary": "No changes.json or visibility candidates yet."}


def copy_crops(changes: dict[str, Any], src_root: Path, viewer: Path) -> None:
    """Copy the crops referenced by `changes` (paths relative to `src_root` or absolute)
    to `viewer/crops/` and point the references there; missing files become null."""
    crops_dir = viewer / "crops"
    shutil.rmtree(crops_dir, ignore_errors=True)
    for change in changes["changes"]:
        crops = change.get("crops") or {}
        for side, rel in crops.items():
            src = src_root / rel if rel else None
            if src is None or not src.is_file():
                print(f"[viewer] warning: crop {rel!r} of change {change.get('id')} not found")
                crops[side] = None
                continue
            crops_dir.mkdir(exist_ok=True)
            shutil.copy2(src, crops_dir / src.name)
            crops[side] = f"crops/{src.name}"


# ------------------------------------------------------------------------ export


def scene_manifest(
    name: str, units: str, clouds: list[Cloud], cam2world: np.ndarray, voxel: float
) -> dict[str, Any]:
    """`scene.json`: file names, statistics and orientation hints for the viewer.

    `bbox_m` is the 1st-99th percentile box of both clouds (robust to stray points),
    `up_axis` the unit vector against gravity and `view_dir` the horizontal direction the
    cameras mostly looked in, all in the world frame of recon.npz."""
    points = np.concatenate([p for p, _ in clouds])
    lo, hi = np.percentile(points, [1, 99], axis=0) if len(points) else np.zeros((2, 3))
    up = estimate_up(cam2world)
    return {
        "name": name,
        "units": units,
        "before_ply": "before.ply",
        "after_ply": "after.ply",
        "changes": "changes.json",
        "stats": {
            "points_before": len(clouds[0][0]),
            "points_after": len(clouds[1][0]),
            "extent_m": (hi - lo).round(3).tolist(),
            "voxel_m": round(voxel, 4),
        },
        "bbox_m": [lo.round(3).tolist(), hi.round(3).tolist()],
        "up_axis": up.round(4).tolist(),
        "view_dir": horizontal_view_dir(cam2world, up).round(4).tolist(),
    }


def export_viewer(out_dir: str | Path, target_points: int = TARGET_POINTS) -> Path:
    """Write the static web viewer for the run in `out_dir` to `<out>/viewer/`."""
    paths = OutputPaths(out_dir)
    viewer = paths.viewer
    viewer.mkdir(parents=True, exist_ok=True)

    recon = load_recon(paths.recon / "recon.npz")
    clouds = [recording_points(recon, r) for r in range(len(RECORDINGS))]
    cam2world = recon["cam2world"]
    del recon  # images and depth maps are no longer needed
    clouds, voxel = downsample_to_target(clouds, target_points)
    for name, (points, colors) in zip(RECORDINGS, clouds):
        write_ply(viewer / f"{name}.ply", points, colors)

    changes = load_changes(paths)
    copy_crops(changes, paths.root, viewer)
    (viewer / "changes.json").write_text(json.dumps(changes, indent=2))

    units = changes.get("scene", {}).get("units", UNITS)
    manifest = scene_manifest(paths.root.resolve().name, units, clouds, cam2world, voxel)
    (viewer / "scene.json").write_text(json.dumps(manifest, indent=2))
    for file in APP_FILES:
        shutil.copy2(VIEWER_SRC / file, viewer / file)

    stats = manifest["stats"]
    print(f"[viewer] {stats['points_before']:,} before / {stats['points_after']:,} after points "
          f"(voxel {100 * voxel:.1f} cm), {len(changes['changes'])} changes -> {viewer}")
    print(f"[viewer] open with: cd {viewer} && python -m http.server 8000")
    return viewer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export the static web viewer of a RoomDiff run to <out_dir>/viewer/.")
    parser.add_argument("out_dir", type=Path, help="run directory containing recon/recon.npz")
    parser.add_argument("--target-points", type=int, default=TARGET_POINTS,
                        help=f"max points per cloud (default {TARGET_POINTS:,})")
    args = parser.parse_args(argv)
    export_viewer(args.out_dir, args.target_points)


if __name__ == "__main__":
    main()
