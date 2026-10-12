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

## M4 instances + changes -> `<out>/instances/`, `<out>/changes.json`, `<out>/crops/`
Implemented (`roomdiff/instances.py`, `roomdiff/changes.py`, model wrappers in
`roomdiff/models.py`; parameters in `configs/default.yaml: instances, changes`).

Instances are built per recording `<rec>` from the M3 blobs occupied in it (disappeared
-> before, appeared -> after, unverified_<rec> -> <rec>): SAM 2.1 masks in up to `n_views`
keyframes, lifted to 3D, fragments merged. Instance ids are 1..N per recording (verified
kinds first, then by point count).

`instances/<rec>.json`: `{recording, image_size [W, H], instances: [...]}`; per instance:
`id`, `recording`, `kind` (disappeared / appeared / unverified), `label` ("object" until
M5), `blob_ids`, `n_points`, `centroid`, `aabb [[min], [max]]`, `extents`, `views` (per
prompted keyframe: `frame` index into recon.npz, `file`, prompt `box`, `mask_box`
[x0,y0,x1,y1] at recon resolution, SAM `score`), `mask_scores`, `mask_score_mean`,
`view_consistency` (mean pairwise cosine of the per-view features, null for one view),
`feature_index` (row in the feature files = id - 1), `n_frame_masks`, `evidence`
(aggregated over the source blobs, verified ones if any: `voxels`, `volume_m3`,
`free_rays` = sum over blob voxels of the other recording's free count, voxel-weighted
`free_frames_mean`, max `observing_views`, summed `occ_points`, `min_camera_distance`).
`instances/<rec>_feats.npy` (N, 1024) float32: L2-normalised DINOv3 ViT-L/16 patch tokens
pooled inside the eroded masks, averaged over views. `instances/<rec>_cls.npy` (N, 1024):
CLS token of the best view's mask box crop. `instances/<rec>_points.npz`: `points` (M, 3)
float32 (1 cm voxels), `colors` (M, 3) uint8, `instance` (M,) int32 id.
`instances/<rec>_masks.npz`: one row per (keyframe, instance) where the instance is
visible (>= `frame_visible_frac` of its points depth-consistent): `keyframe` (file name),
`instance_id`, `score` (SAM), `rle` (COCO RLE counts string), `size` [H, W] (recon
resolution); read with `roomdiff.instances.load_masks` / `decode_rle`.
`instances/meta.json`: parameters, selected / dropped blobs with the reason (small or
too large unverified, hidden surface, no clear view, no mask, too few points), fragment
and instance counts, runtime, peak VRAM. `instances/matching.json`: cosine and cost
matrices of the moved candidates and the accepted pairs.

`changes.json`:
```json
{
  "scene": {"before": "...mov", "after": "...mov", "units": "metres (approx.)"},
  "changes": [
    {"id": 1, "type": "moved|added|removed|unverified", "label": "office chair",
     "confidence": 0.86,
     "before": {"instance_id": 3, "centroid": [x,y,z], "aabb": [[x0,y0,z0],[x1,y1,z1]],
                "extents": [dx,dy,dz], "n_points": 5120, "blob_ids": [9, 14]},
     "after":  {"instance_id": 7, "centroid": [x,y,z], "aabb": [[...],[...]], ...},
     "transform": {"translation_m": 0.92, "rotation_deg": 31.0, "R": [[...]], "t": [...],
                   "icp_fitness": 0.82, "icp_rmse": 0.012},
     "evidence": {"free_rays": 412, "observing_views": 9, "appearance_similarity": 0.81,
                  "vlm_verdict": "same object, moved"},
     "description": "The office chair moved about 0.9 m from the desk towards the window.",
     "crops": {"before": "crops/1_before.jpg", "after": "crops/1_after.jpg"}}
  ],
  "summary": "3 changes detected: 1 moved, 1 added, 1 removed; 1 region unverified."
}
```
Every change has all keys; `before` / `after` is null on the side where the object does
not exist (removed: after, added: before, unverified: the recording that never observed
it) and `transform` is null unless moved. `R`, `t` map before points to after points
(`x_after = R x_before + t`); `translation_m` = displacement of the before centroid;
`rotation_deg` is null when the ICP fit is not trusted (symmetric 3 cm fitness below
`icp_min_fitness`). Order: moved, removed, added, unverified; by confidence within a type.
M4 writes `label: "object"`, a templated `description` and `vlm_verdict: null` (M5 fills
them). Evidence keys: removed / added / unverified: `free_rays`, `free_frames_mean`,
`observing_views` (keyframes of the other recording that saw through the place),
`occ_points`, `mask_score`, `view_consistency` (+ `seen_in` for unverified); moved:
`free_rays` (both places), `observing_views` (mean), `observing_views_before_place`,
`observing_views_after_place`, `appearance_similarity` (DINOv3 cosine), `cls_similarity`,
`mask_score`, `icp_fitness`.
Confidence = (w_mask S_mask + w_free S_free + w_consistency S_cons) / (sum of weights):
S_mask = mean SAM score, S_free = min(observing_views / free_views_full, 1) (moved: mean of
both places), S_cons = view consistency (0.5 if unknown; moved: mean of both);
unverified: 0.5 (w_mask S_mask + w_consistency S_cons) / (w_mask + w_consistency) <= 0.5.
Crops (paths relative to `<out>`; JPEG from the full-resolution keyframes, object box
x `crop_context`, at least `crop_min_px` per side): `<id>_<rec>.jpg` shows the object in its
own recording with mask outline and box; the other side shows the other recording looking
at the same 3D box (the empty place or the new object). Moved objects have both object
crops plus `<id>_before_place.jpg` / `<id>_after_place.jpg` (the other recording at the old
/ new place). A side is missing if no keyframe of that recording sees the box.
`changes.rrd` / `scene.rrd`: boxes per change coloured by type, instance points, moved
arrows (`roomdiff/viz_rerun.py`).

## M5/M6 outputs
`report.md` (summary + per-change sections with crops), `scene.rrd` (Rerun; written by
`roomdiff viz`: reconstruction, M3 candidate blobs, M4 changes),
`viewer/` (static web viewer assets: downsampled clouds, boxes, crops, changes.json).
