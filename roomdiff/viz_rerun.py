"""Rerun recordings of the reconstruction (M2) and the change candidates (M3).

Entities: `world/<rec>/points` (valid points coloured from the keyframes) and
`world/<rec>/cam/<i>` (pose + pinhole, keyframe image for a few cameras) for
`<rec>` in before / after; `world/changes/<type>/<id>` (one box per candidate blob,
labelled "id:type") and `world/changes/<type>/voxels` (subsampled voxel centres).

Files: `<out>/scene.rrd` (everything), `<out>/recon/recon.rrd` (reconstruction only),
`<out>/visibility/visibility.rrd` (candidates only). Open with `rerun <file>`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from roomdiff.paths import RECORDINGS, OutputPaths
from roomdiff.recon import load_recon, recording_points
from roomdiff.visibility import CANDIDATE_TYPES, TYPE_COLORS

if TYPE_CHECKING:
    import rerun as rr

CAMERA_COLORS = {"before": (66, 133, 244), "after": (234, 67, 53)}  # blue / red frusta
IMAGES_PER_RECORDING = 6  # cameras that also get their keyframe image
POINT_RADIUS = 0.004  # m
MAX_VOXELS_PER_TYPE = 50_000  # candidate voxel centres logged per type


def _open_stream(rrd: Path) -> rr.RecordingStream:
    """New recording saved to `rrd`, with OpenCV world axes (x right, y down)."""
    import rerun as rr

    stream = rr.RecordingStream("roomdiff")
    stream.save(rrd)
    stream.log("world", rr.ViewCoordinates.RDF, static=True)
    return stream


def _log_recon_entities(stream: rr.RecordingStream, out_dir: Path, max_points: int) -> int:
    """Log clouds and cameras of `<out>/recon/recon.npz`; returns the camera count."""
    import rerun as rr

    recon = load_recon(OutputPaths(out_dir).recon / "recon.npz")
    rng = np.random.default_rng(0)
    height, width = recon["depth"].shape[1:]
    for rec_id, name in enumerate(RECORDINGS):
        points, colors = recording_points(recon, rec_id)
        if len(points) > max_points // 2:
            keep = rng.choice(len(points), max_points // 2, replace=False)
            points, colors = points[keep], colors[keep]
        stream.log(
            f"world/{name}/points",
            rr.Points3D(points, colors=colors, radii=POINT_RADIUS),
            static=True,
        )

        frames = np.flatnonzero(recon["rec"] == rec_id)
        count = min(IMAGES_PER_RECORDING, len(frames))
        with_image = set(np.linspace(0, len(frames) - 1, count).round().astype(int).tolist())
        for j, i in enumerate(frames):
            entity = f"world/{name}/cam/{j}"
            T = recon["cam2world"][i]
            stream.log(entity, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]),
                       static=True)
            stream.log(
                entity,
                rr.Pinhole(
                    image_from_camera=recon["K"][i],
                    resolution=[width, height],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.15,
                    color=CAMERA_COLORS[name],
                ),
                static=True,
            )
            if j in with_image:
                image = rr.Image(recon["images"][i]).compress(jpeg_quality=85)
                stream.log(f"{entity}/image", image, static=True)
    return len(recon["rec"])


def _log_visibility_entities(stream: rr.RecordingStream, out_dir: Path,
                             max_voxels: int = MAX_VOXELS_PER_TYPE) -> int:
    """Log the candidate blobs of `<out>/visibility/` as boxes plus voxel centres;
    returns the number of blobs."""
    import rerun as rr

    folder = OutputPaths(out_dir).visibility
    blobs = json.loads((folder / "candidates.json").read_text())["candidates"]
    for blob in blobs:
        lo, hi = np.asarray(blob["aabb"], np.float64)
        stream.log(
            f"world/changes/{blob['type']}/{blob['id']}",
            rr.Boxes3D(centers=[(lo + hi) / 2], half_sizes=[(hi - lo) / 2],
                       labels=[f"{blob['id']}:{blob['type']}"],
                       colors=[TYPE_COLORS[blob["type"]]]),
            static=True,
        )
    rng = np.random.default_rng(0)
    with np.load(folder / "grid.npz") as grid:  # read only the arrays needed here
        origin, voxel = grid["origin"], float(grid["voxel_size"])
        for kind in CANDIDATE_TYPES:
            ijk = np.argwhere(grid[f"labels_{kind}"] > 0)
            if len(ijk) == 0:
                continue
            if len(ijk) > max_voxels:
                ijk = ijk[rng.choice(len(ijk), max_voxels, replace=False)]
            stream.log(
                f"world/changes/{kind}/voxels",
                rr.Points3D(origin + (ijk + 0.5) * voxel, colors=TYPE_COLORS[kind],
                            radii=voxel / 2),
                static=True,
            )
    return len(blobs)


def log_recon(out_dir: str | Path, max_points: int = 2_000_000) -> Path:
    """Write `<out>/recon/recon.rrd` from `<out>/recon/recon.npz`; returns its path.

    At most `max_points` points in total (random subset, half per recording).
    """
    rrd = OutputPaths(out_dir).recon / "recon.rrd"
    stream = _open_stream(rrd)
    cameras = _log_recon_entities(stream, Path(out_dir), max_points)
    stream.flush()
    print(f"[viz] wrote {rrd} ({cameras} cameras)")
    return rrd


def log_visibility(out_dir: str | Path) -> Path:
    """Write `<out>/visibility/visibility.rrd` (candidate blobs only); returns its path."""
    rrd = OutputPaths(out_dir).visibility / "visibility.rrd"
    stream = _open_stream(rrd)
    blobs = _log_visibility_entities(stream, Path(out_dir))
    stream.flush()
    print(f"[viz] wrote {rrd} ({blobs} candidate blobs)")
    return rrd


def log_scene(out_dir: str | Path, max_points: int = 2_000_000) -> Path:
    """Write `<out>/scene.rrd`: the reconstruction plus, if M3 has run, its candidate
    blobs, in one recording; returns its path."""
    out_dir = Path(out_dir)
    rrd = out_dir / "scene.rrd"
    stream = _open_stream(rrd)
    cameras = _log_recon_entities(stream, out_dir, max_points)
    blobs = None
    if (OutputPaths(out_dir).visibility / "candidates.json").exists():
        blobs = _log_visibility_entities(stream, out_dir)
    stream.flush()
    extra = f", {blobs} candidate blobs" if blobs is not None else ", no visibility output"
    print(f"[viz] wrote {rrd} ({cameras} cameras{extra})")
    return rrd
