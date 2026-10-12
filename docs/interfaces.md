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
them, see below). Evidence keys: removed / added / unverified: `free_rays`, `free_frames_mean`,
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

## M5 semantics -> `<out>/changes.json` (rewritten), `<out>/semantics/`, `<out>/report.md`
Implemented (`roomdiff/semantics.py`, `roomdiff/vlm.py`, `roomdiff/report.py`, command
`roomdiff describe`; parameters in `configs/default.yaml: semantics`; the prompts are
constants in `roomdiff/semantics.py`). Qwen3-VL-8B-Instruct, bf16, greedy decoding, every
crop resized to <= `image_max_side` px. The first run backs the M4 document up to
`semantics/changes_m4.json`; later runs start from that backup (re-run `roomdiff changes`
to refresh it). Steps: naming -> re-pairing -> judge -> descriptions -> summary.

Changes keep the M4 schema and ids (stable, so ids have gaps where entries were rejected
or merged); M5 sets or adds:
* `label`: VLM name, normalised (lowercase, no article, cut at a preposition, <= 3 words);
  `attributes` (colour / material), `object_description` (the VLM's sentence about the
  object).
* `evidence.vlm_verdict` "confirmed" | "unclear" (rejected entries move to `rejected`),
  `evidence.vlm_reason`; unclear: `confidence` x `unclear_factor`. Moved: `motion_kind`
  "rigid" | "deformed" | "unknown" and `evidence.vlm_motion_reason` (`motion_kind` is null
  for the other types). Unverified regions are rejected unless the VLM sees a distinct
  object (not wall / floor / door frame / window / mirror reflection / shadow / lighting).
* Re-paired removed R + added or after-only unverified A (matching labels: same label,
  head noun or synonym group; centroids < `repair_max_distance`; VLM: same object) -> one
  `moved` change with R's id: `before` = R's side, `after` = A's, `transform` from the
  centroids (`R` = I, `rotation_deg` / `icp_*` null), confidence = mean, `crops`
  `before` = R's, `after` = A's, `before_place` = R's after view, `after_place` = A's
  before view, `evidence.merged_from` [R, A] and `merged_types`.
* A moved change M absorbing an added change P (matching labels, gap between the after
  boxes <= `merge_max_distance`, VLM: parts of one object, e.g. a spread-out quilt):
  `after` = union box (`instance_ids` [M's, P's], `n_points` summed, point-weighted
  centroid), transform = centroid translation (rotation null), confidence = mean,
  `evidence.merged_from` [M, P], `evidence.merged_part_reason`, `crops.after_part`.
* `description`: one sentence (label, type, distance in m with one decimal, size, coarse
  place "far left part of the room" along the mean camera direction, height of the lowest
  observed part above the floor = median of the lowest 5 % of all valid points, evidence
  counts), polished by one text-only VLM call if every number and the label survived;
  `description_template` keeps the template sentence.

Top level: `rejected` (same schema, `vlm_verdict` "rejected" + reason), `summary` =
"N changes: a moved, b added, c removed; d unverified." + a VLM paragraph.
`semantics/meta.json`: model, params, runtime per step, VLM calls and seconds, counts in /
out, rejected (id, type, label, reason), merges (every VLM pair check: ids, labels,
distance or box gap, verdict, reason), verdict per id, polished / template-kept counts,
scene frame (up vector, floor height), `peak_vram_gb`, `peak_ram_gb`.
`semantics/log.jsonl`: one line per VLM call `{tag, images, prompt, answer, seconds}`.
`report.md`: title, summary, counts table, one section per change (type badge, label,
confidence, verdict, motion, description, evidence line, crops as relative markdown
images), collapsible rejected list with the VLM reasons, coverage note from
`recon/meta.json`.

## M6 outputs
`scene.rrd` (Rerun; written by `roomdiff viz`: reconstruction, M3 candidate blobs,
changes), `viewer/` (`roomdiff viewer`: static web viewer assets: downsampled clouds,
boxes, crops, changes.json).

## Evaluation (SceneDiff) -> `<out-root>/`
Implemented (`roomdiff/eval_scenediff.py`, `roomdiff eval`, parameters in
`configs/default.yaml: eval`). Scoring is done by the official, unmodified
`third_party/scene_diff/scripts/evaluate_multiview.py` (MIT).

Layout: `runs/<pair>/` (one `roomdiff run` per pair with `ingest.mode=fixed`, GPU-locked
through `scripts/run_guarded.sh -g`; pairs that already have `changes.json` are skipped),
`logs/<pair>.log`, `pred/<pair>/object_masks.pkl` + `prediction.json` (objects per type,
masks per video, mapped benchmark frames, mask / GT / target sizes, aspect check),
`pred/result_frame_dup_1_obj_dup_1.txt` (the evaluator appends the suffix),
`pred/evaluator_log.txt`, `pred/summary.json` (`metrics`: `per_view_ap` = obj/im AP,
`per_scene_ap` = obj/sc AP, `per_scene_ap_type` = obj/sc AP with type, `global_iou` =
px/im IoU with pixel TP/FP/FN, object TP/FP/FN, `scenes`; plus `not_predicted`,
`evaluator_errors` and the per-pair prediction stats).

Pairs: folder names under `<eval.data_root>/data/`; `before` = `original_video1.*`,
`after` = `original_video2.*` (raw phone videos; the `video{1,2}.mp4` re-encodes carry
painted mask overlays and are never used). `write_prediction` refuses a run whose
`keyframes/index.json` has the videos the other way round.

Frames: the evaluator keeps GT frames whose index in the 30 fps re-encode is a multiple of
30 and renumbers them k = index // 30, so GT frame k is time k s of the original video
(time, not index: the originals run at e.g. 29.92 fps). Fixed ingest mode
(`ingest.fixed_interval_s`, default 1.0) takes the decoded frame closest to every
t = k * interval (no sharpness / motion filtering) and adds `t_target` and
`benchmark_frame` (= round(t_target) when t_target is a whole second, else null) to each
keyframe entry of `keyframes/index.json`. Sharp-mode keyframes (no `benchmark_frame`) map
to k = round(timestamp) only if |timestamp - k| < `eval.time_tol_s` (0.1 s). Keyframes
mapped to the same k: the closest in time is used.

`object_masks.pkl` (pickle protocol 4, one per pair):
```
{"H": int, "W": int,                     # GT size scaled to max side 1024 (upright)
 <change id>: {"video_1": {k: {"mask": {"size": [h, w], "counts": str}, "cost": float}},
               "video_2": {...},
               "label": 1 | 0}}          # 1 = moved, 0 = added / removed / unverified
```
One object per change of `changes.json` (`rejected` entries are not used): the
before-instance masks of `instances/before_masks.npz` at mapped keyframes go to `video_1`,
the after-instance masks to `video_2`; moved changes keep both keys (the evaluator derives
the type from their presence), others have one. Unverified changes are included when
`eval.include_unverified` (default true; their confidence is <= 0.5, so they rank after
the verified changes and can only add recall to AP, but their pixels count in the global
IoU). `cost` = change confidence x SAM mask score (the evaluator ranks per-view
detections by it and objects by its mean). Changes without a mask at any mapped frame are
left out. Masks are COCO RLE at the reconstruction resolution (upright, same aspect as
the GT, e.g. 672 x 378 for 1920 x 1080 portrait GT; checked in `prediction.json:
aspect_ok`); the evaluator resizes them with nearest neighbour. A failed run gets an
empty prediction (`{"H", "W"}` only) so its GT objects count as misses: the evaluator
silently drops scenes without `object_masks.pkl`.

Evaluator quirks: (1) it imports `faiss` (not installed) through `utils.py`, used only for
`--visualize True`; the launcher stubs that module. (2) It reads the split lists from the
cwd-relative path `data/scenediff_benchmark/splits/`; it is run in `pred/_evaluator/` with
split files listing exactly the requested pairs (`--splits val --sets all`). (3) It loads
`segments.pkl` with plain `pickle.load`; it is given data-only copies
(`pred/_evaluator/gt/`) made with a no-globals unpickler. (4) `get_target_hw_from_gt`
reads the COCO size `[H, W]` as `(W, H)`, so portrait GT is scored on a landscape canvas
(1920 x 1080 -> 576 x 1024); GT and predictions are stretched alike. (5) A scene that
fails to load is printed as `Error loading scene ...` and skipped but still counted in
`Scenes:`; such lines are collected in `summary.json: evaluator_errors`.
