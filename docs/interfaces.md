# Module interfaces (data contracts between pipeline stages)

All stages read/write under one output directory `<out>` (see `roomdiff/paths.py`).
Coordinates: OpenCV camera convention (x right, y down, z forward); world frame = the
shared frame produced by the joint reconstruction; units = metres as estimated by the
reconstruction model (approximately metric; a `scale_factor` override exists).

## M1 ingest -> `<out>/keyframes/`
`before/*.jpg`, `after/*.jpg`, `index.json` (per recording: video info, per-keyframe file,
source frame index, timestamp, sharpness, size). Implemented.

## M2 recon -> `<out>/recon/`
`recon.npz` with keyframes of both recordings concatenated (before first, then after):
| key | shape / type | meaning |
|---|---|---|
| `rec` | (N,) uint8 | 0 = before, 1 = after |
| `files` | (N,) str | keyframe file names |
| `images` | (N,H,W,3) uint8 | images at the model resolution (multiple of 14) |
| `cam2world` | (N,4,4) float32 | camera-to-world, metric |
| `K` | (N,3,3) float32 | intrinsics at the model resolution |
| `depth` | (N,H,W) float32 | z-depth in the camera frame, metric |
| `conf` | (N,H,W) float16 | confidence probability in [0,1] |
| `valid` | (N,H,W) bool | conf > threshold and not a depth/normal edge |
| `metric_scale` | float | scale factor reported by the model |
`meta.json`: model name, resolution, frame counts, runtime, VRAM, alignment diagnostics
(median nearest-neighbour distance of confident before-points to the after cloud and vice
versa, fraction under 3 cm / 10 cm), and the chosen `scale_factor`.
`before.ply`, `after.ply`: voxel-downsampled coloured clouds (viewer / debugging).
World points are recomputed from `depth`, `K`, `cam2world` (helper in `roomdiff/geometry.py`).

## M3 visibility -> `<out>/visibility/`
Implemented (`roomdiff/visibility.py`, parameters in `configs/default.yaml: visibility`).
Voxel (i, j, k) spans `origin + [i, i+1) * voxel_size` along x (j: y, k: z); arrays are
3D with shape `dims` (C order).

`grid.npz` (`load_grid`): `voxel_size`, `origin` (3,), `dims` (3,); per recording
`<rec>` in before / after: `state_<rec>` uint8 (0 unknown, 1 free, 2 occupied),
`free_count_<rec>` int16 (keyframes that saw the voxel empty), `occ_count_<rec>` int32
(valid points inside); `labels_<type>` int32 per candidate type (0 = none, else the blob
id of `candidates.json`).

`candidates.json`: `{voxel_size, origin, dims, counts, candidates: [...]}`, blobs sorted by
volume, ids 1..N unique across types. Types (own recording = where it is occupied):
`disappeared` = occupied(before) & free(after), `appeared` = occupied(after) &
free(before), `unverified_before` / `unverified_after` = occupied in that recording and
unknown in the other; voxels within `dist_margin` of the other recording's surfaces are
excluded. Per blob: `id, type, voxels, volume_m3, centroid, aabb [[min], [max]]`,
`evidence {free_frames_mean` (mean free_count of the other recording over the blob),
`occ_points` (points of the own recording), `observing_views` (keyframes of the other
recording that saw past the centroid), `min_camera_distance` (centroid to the nearest
camera of the other recording)`}`, `views {before, after}`: best keyframes of the centroid
(`frame` index into recon.npz, `file`, `occluded`, projected `box` [x0,y0,x1,y1],
`crop` = `debug/blob_<id>_<type>_<rec>_<k>.jpg`).

`meta.json`: parameters used, voxel size source, grid, voxels per state, candidate voxel
counts (raw / after opening / in blobs), blobs per type, runtime, peak VRAM.
`visibility.rrd`: blobs as Rerun boxes. Projection helpers for later stages:
`roomdiff/projection.py` (`project_to_pixels`, `project_aabb`, `best_views`).

## M4 instances + changes -> `<out>/instances/`, `<out>/changes.json`
`instances/<rec>.json`: per object: id, label votes, point count, centroid, aabb, best
views (keyframe + 2D box), feature vector index. `instances/<rec>_masks.npz`: per-keyframe
RLE masks with object ids. `instances/<rec>_feats.npy`: L2-normalised features.

`changes.json`:
```json
{
  "scene": {"before": "...mov", "after": "...mov", "units": "metres (approx.)"},
  "changes": [
    {"id": 1, "type": "moved|added|removed|unverified", "label": "office chair",
     "confidence": 0.86,
     "before": {"instance_id": 3, "centroid": [x,y,z], "aabb": [[x0,y0,z0],[x1,y1,z1]]},
     "after":  {"instance_id": 7, "centroid": [x,y,z], "aabb": [[...],[...]]},
     "transform": {"translation_m": 0.92, "rotation_deg": 31.0, "R": [[...]], "t": [...]},
     "evidence": {"free_rays": 412, "observing_views": 9, "appearance_similarity": 0.81,
                  "vlm_verdict": "same object, moved"},
     "description": "The office chair moved about 0.9 m from the desk towards the window.",
     "crops": {"before": "crops/1_before.jpg", "after": "crops/1_after.jpg"}}
  ],
  "summary": "3 changes detected: 1 moved, 1 added, 1 removed; 1 region unverified."
}
```

## M5/M6 outputs
`report.md` (summary + per-change sections with crops), `scene.rrd` (Rerun; written by
`roomdiff viz` since M3: reconstruction + candidate blobs),
`viewer/` (static web viewer assets: downsampled clouds, boxes, crops, changes.json).
