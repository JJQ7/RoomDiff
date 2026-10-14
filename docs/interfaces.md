# Module interfaces (data contracts between pipeline stages)

All stages read/write under one output directory `<out>` (see `roomdiff/paths.py`).
Coordinates: OpenCV camera convention (x right, y down, z forward); world frame = the
shared frame produced by the joint reconstruction, levelled by M2 (`recon.level_world`,
default on; `levelled` in `recon/meta.json`): +Y up (against gravity), the floor plane at
y = 0 and the first before-camera's optical axis projected onto +Z (so +X is to its left),
then, with `recon.snap_to_walls` (default off) and a clear wall direction, turned about +Y
so that the dominant wall direction lies along +X / +Z (`walls_snapped`; the first camera
then looks within 45 degrees of +Z). Without levelling the frame is Pi3X's (roughly the
first camera). Units = metres as
estimated by the reconstruction model (approximately metric; a `scale_factor` override
exists).

## M1 ingest -> `<out>/keyframes/`
`before/*.jpg`, `after/*.jpg`, `index.json` (per recording: video info, per-keyframe file,
source frame index, timestamp, sharpness, size). Per recording `square_crop` (int) is 0,
or the side of the central square the keyframes of both recordings were cropped to when
one video is portrait and the other landscape (square matches either; one Pi3X pass needs
one image size): the per-keyframe sizes are those of the cropped JPEGs while `video` still
describes the source video. Implemented.

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
versa, fraction under 3 cm / 10 cm), and the chosen `scale_factor`. Levelling (after the
joint pass or the separate-mode alignment, before `scale_factor`): `levelled` (bool);
when true also `up_axis_original` (the up direction in the model frame that became +Y),
`floor_inliers_fraction` (share of the floor candidates within 2 cm of the fitted floor
plane, null if no plane was found) and `level` {`method` ("ransac floor plane" | "camera
up (no floor plane)"), `camera_up` (initial estimate: mean camera down axis without its
components along the camera x axes), `floor_vs_camera_up_deg`, `floor_offset_m`,
`points_used`, `wall_votes`, `transform` (4x4, model frame -> levelled frame)}; with
`snap_to_walls` also `walls_snapped` (bool), `wall_yaw_deg` (dominant wall direction in
the camera-levelled frame, in (-45, 45], `rotation_y` / `obb.yaw_deg` convention; the
snapped frame is that one turned by `rotation_y(-wall_yaw_deg)`) and `wall_peak_fraction`
(share of the wall votes in the histogram peak; snapped only if >= `wall_peak_fraction`
of the config and >= 50 votes; null without votes). Walls: valid points (every 4th
pixel) 0.3-2.0 m above the floor, 5 cm voxels, normals by local PCA over 16 neighbours;
planar neighbourhoods (smallest eigenvalue < 0.1 x the middle one) with a normal within
15 degrees of horizontal vote with the normal's yaw modulo 90 degrees into 1-degree bins;
peak = the +-5 degree window with the most votes, yaw = circular mean of its votes. Up = camera estimate
refined by the floor plane: MSAC (RANSAC with truncated squared residuals, 2 cm cap;
points clearly below a candidate plane cost double) on the lowest 10 % of the valid
points (every 4th pixel) along the camera up, planes within 20 degrees of it only, then
least squares on the inliers; no such plane -> the camera up and the median of the lowest
5 % of the points as floor height.
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
camera of the other recording)`}`, `views {before, after}`: the `debug_views` best
keyframes of the blob's voxel centres by the view scorer below (own recording: voxels
depth-consistent within `occlusion_tol`; other recording: not occluded) (`frame` index
into recon.npz, `file`, `occluded` (fewer than 15 % of the voxels visible),
`visible_fraction`, `box` [x0,y0,x1,y1] = 2D box of the projected voxel centres,
`crop` = `debug/blob_<id>_<type>_<rec>_<k>.jpg`).

View scorer (`roomdiff.projection.score_views`, used for the M3 debug crops, the M4 SAM
prompting views and the M4 crops): per keyframe, the object's points (blob voxel centres
or instance points; a fixed subset of at most 2000) are projected; a point is visible if
it lands on a valid pixel whose depth is within the tolerance of its own (object views),
or not more than the tolerance in front of it (views of the place in the other
recording). Score = area of the 2D box of the visible points x visible fraction x 0.8
if the projected object touches the image border x (1 - 0.2 e cos^2 a), e = elongation of
the object's footprint (1 - minor / major variance of x-z), a = angle between the
horizontal viewing ray and its long horizontal axis. Everything is measured in the
images, so the ranking does not change when the world frame is turned about +Y
(`tests/test_visibility.py`); the previous ranking by the projected axis-aligned box did.

`meta.json`: parameters used, voxel size source, grid, voxels per state, candidate voxel
counts (raw / after opening / in blobs), blobs per type, runtime, peak VRAM.
`visibility.rrd`: blobs as Rerun boxes. Projection helpers for later stages:
`roomdiff/projection.py` (`project_to_pixels`, `project_aabb`, `score_views`, `best_views`).

## Appearance channel -> `<out>/appearance/`
Implemented (`roomdiff/appearance.py`, command `roomdiff appearance`, run between
visibility and instances; parameters in `configs/default.yaml: appearance`, models / device
/ `lift_voxel` from `instances`; `enabled: false` skips the stage and the instance stage
then ignores any old output). Finds changes that M3 cannot see because they lie within
`visibility.dist_margin` of a surface (a wallet on a table, a jumper on a bed, an object
replaced in place):

1. Co-visible pairs: directional share of a stride-`covis_stride` (8) subsample of keyframe
   i's valid pixels that projects inside keyframe j onto a valid pixel with |z - depth_j| <
   max(`covis_depth_abs`, `covis_depth_rel` z) (0.05 m, 0.03); score = min of both
   directions; per keyframe the `pairs_per_frame` (2) best partners of the other recording
   with score >= `min_covisibility` (0.3), for before -> after and after -> before.
2. DINOv3 ViT-L/16 patch tokens of every keyframe (the `Featurizer.patch_tokens` grid,
   24 x 42 at 384 x 672 for a 378 x 672 image), L2-normalised.
3. Per pair i -> j, each patch centre of i with valid depth is reprojected into j
   (u, v, z). *Observed*: inside j, valid pixel, depth_j(u, v) > z - tol (tol as in 1; a
   point behind j's surface or outside j gets no vote). On observed patches `feat` = 1 -
   cos(F_i(p), F_j(u, v)) (bilinear in j's grid, renormalised) and `geom` = 1 if
   depth_j(u, v) - z > `geom_tol_abs` + `geom_tol_rel` z (0.03 + 0.02 z: j sees through
   the place, the surface seen in i is gone; asymmetric, an object that appeared in j is
   found from j's side).
4. Per keyframe: score = mean over the observing partners of (geom + `w_feat` feat), 0
   with fewer than `min_votes` (2) observing partners; bilinearly upsampled, thresholded
   at `score_threshold` (0.60; `threshold_mode: max_entropy` uses Kapur's threshold of
   all scores instead, always reported); 8-connected regions of `min_region_frac` (0.2 %)
   .. `max_region_frac` (25 %) of the image, at most `max_regions_per_frame` (10) by
   mean score.
5. Per region: SAM 2.1 prompted with the region box enlarged by `box_enlarge` and up to
   `n_prompt_points` (5) highest-scoring patch centres inside it; the best-scoring mask
   with IoU > `min_mask_iou` (0.3) with the region; its valid-depth pixels lifted (1 cm
   voxels, outliers removed), >= `min_points` (30). Static-surface rule: a candidate
   whose mean feature term is at most half its score (depth-driven) and whose points lie
   more than `static_frac` (50 %) within `static_dist` (2 cm) of the other recording's
   points (1 cm voxels) is dropped; feature-driven candidates are kept. Candidates of one
   recording are clustered (union-find: more than `cluster_frac` (0.3) of the points of
   one within `cluster_dist` (3 cm) of the other); clusters need candidates in
   `min_frames` (2) keyframes. Cluster points = union of the candidates' points, minus
   those that fewer than half of the cluster's keyframes seeing them (depth-consistent, or
   inside that keyframe's mask) put inside their mask (voted only with >= 3 keyframes).

Files: `feats_<rec>.npy` (n, gh, gw, D) float16 tokens of the recording's keyframes in
recon order + `feats_<rec>.json` (model, files, image size: the cache is reused only if
all match); `pairs.json` (`frames` per recording, `covisibility` (n_before, n_after),
`partners` per recording: keyframe -> partner keyframes, recon indices);
`scores_<rec>.npz` (`frames` (n,), `partners` (n, pairs_per_frame) -1 padded, `score`,
`feat` (w_feat x mean dissimilarity), `geom` (share of see-through votes) (n, gh, gw)
float16, `votes` uint8, `threshold`); `regions.json` (per region: recording, frame, file,
box, area, mean score / feat / geom, partners, SAM score, mask IoU, static fraction,
`outcome`: candidate / no SAM mask overlapping the region / too few points / static
surface); `clusters.json` (`image_size`, `threshold`, `clusters`: `id` (1..N over both
recordings), `recording`, `kind` (disappeared for before, appeared for after),
`candidates`, `n_points`, `consensus_dropped`, `centroid`, `aabb`, `extents`, `obb`,
`views` (per keyframe: `frame`, `file`, prompt `box`, `mask_box`, SAM `score`,
`region_score`, `feature_share`), `evidence` {`frames_supporting`, `mean_score`,
`feat_term`, `geom_term` (means over the cluster's regions), `partners_visible` (distinct
partner keyframes that observed its regions), `observing_views` (keyframes of the other
recording seeing >= 15 % of the points unoccluded, 5 cm), `free_rays` / `free_frames_mean`
(other recording's M3 free counts summed / averaged over the voxels holding the points),
`min_camera_distance`, `static_fraction`}); `clusters_points.npz` (`points`, `colors`,
`cluster`); `clusters_masks.npz` (`cluster`, `frame`, `rle` (COCO counts), `size`);
`meta.json` (params, models, token grid, cache hits, keyframes with partners, threshold
{used, fixed, max_entropy, mode}, score stats, regions / candidates / clusters per
recording, region outcomes, dropped clusters, runtime, `peak_vram_gb`, `peak_ram_gb`).

## M4 instances + changes -> `<out>/instances/`, `<out>/changes.json`, `<out>/crops/`
Implemented (`roomdiff/instances.py`, `roomdiff/changes.py`, model wrappers in
`roomdiff/models.py`; parameters in `configs/default.yaml: instances, changes`).

Instances are built per recording `<rec>` from the M3 blobs occupied in it (disappeared
-> before, appeared -> after, unverified_<rec> -> <rec>): SAM 2.1 masks in up to `n_views`
keyframes (best by the view scorer on the blob voxels, at least 15 % of them visible,
viewing directions `min_view_angle` apart; prompt = box of the projected voxels enlarged
by `box_enlarge` + up to `n_prompt_points` depth-visible voxels; the best-scoring mask
with >= 50 % of its pixels in the box; up to `mask_grow_rounds` (3) times, a mask that
reaches a side of the prompt box not on the image border (pixels within 2 px spanning
>= 25 % of the mask's extent along it) is re-prompted with the box extended by its own
size on those sides, keeping the new mask only if it contains >= 80 % of the old one:
the object continues beyond the blob, e.g. the top of a ladder whose blob is its lower
half; the view's `box` is the final prompt box), lifted to 3D per view:
all valid-depth mask pixels, binned into `lift_cell_m` (1.5 cm) cells linked when at most
`lift_link_m` (3 cm) apart; the connected components holding at least 10 % of the blob
voxels that touch a cell (within half a voxel diagonal + one cell) are kept, minus points
on static surfaces (`drop_static`: inside a voxel, dilated by one, that the other
recording occupies) unless within `near_voxel_m` of a blob voxel. If the outlier-filtered
rest spans an oriented box more than `max_growth` (6) times the blob's (`floor_obb` of the
voxel centres grown by one voxel on every side: M3 trims blobs near the other recording's
surfaces), misses the blob or is empty, that view falls back to
the bounded rule (points in the blob box expanded by `expand_m` and not static, or within
`near_voxel_m` of a blob voxel) and the instance gets `lift_fallback` true. Then the
fragments are merged: first by overlap (mask IoU > `merge_mask_iou`
in a shared keyframe, or more than `merge_point_frac` of one's points within
`merge_point_dist` of the other), then touching fragments of the same kind (a chair seat
and its base): at least `merge_gap_frac` of the smaller one's points within `merge_gap_m`
(0.05 m) of the other's, if no side of the union's oriented box reaches
`merge_max_extent_m` (1.5 m); greedy, largest near fraction first. Instance ids are 1..N
per recording (verified kinds first, then by point count). Then the appearance clusters
(with `appearance.enabled` and `<out>/appearance/clusters.json`) are appended: a cluster
with more than `appearance.covered_frac` (50 %) of its points within
`appearance.covered_dist` (3 cm) of the points of one instance of its recording is
already covered (listed in meta.json), the others become instances with ids N+1.. (by
point count), `source` "appearance", kind disappeared (before) / appeared (after), the
cluster's views and masks, no blobs.

`instances/<rec>.json`: `{recording, image_size [W, H], image_size_full [W, H] (keyframe
size of the `rle_full` masks below, null if there are none), instances: [...]}`; per instance:
`id`, `recording`, `kind` (disappeared / appeared / unverified), `label` ("object" until
M5), `blob_ids`, `n_points`, `centroid`, `aabb [[min], [max]]`, `extents`, `obb`
(oriented box with a vertical axis: `center` [x, y, z], `extents` [w, h, d] with w / d
along the box's own x / z axes and h along y, `yaw_deg` in (-45, 45]: the box is an
axis-aligned box of these extents rotated about +Y by `yaw_deg`, three.js `rotation.y`
convention, i.e. its x axis points along (cos, 0, -sin); footprint = minimum-area
rectangle of the points projected on the x-z (floor) plane, height = their y range; it is
gravity-aligned when the recon frame is levelled), `views` (per
prompted keyframe: `frame` index into recon.npz, `file`, prompt `box`, `mask_box`
[x0,y0,x1,y1] at recon resolution, SAM `score`), `mask_scores`, `mask_score_mean`,
`view_consistency` (mean pairwise cosine of the per-view features, null for one view),
`feature_index` (row in the feature files = id - 1), `n_frame_masks`, `merged_from`
(blob ids of each part joined by the gap rule, e.g. [[7, 18], [11]]; empty if none),
`lift_fallback` (some view of a part used the bounded lifting), `source` ("geometry" for
M3 blobs, "appearance" for appearance clusters), `evidence`
(aggregated over the source blobs, verified ones if any: `voxels`, `volume_m3`,
`free_rays` = sum over blob voxels of the other recording's free count, voxel-weighted
`free_frames_mean`, max `observing_views`, summed `occ_points`, `min_camera_distance`;
appearance instances: `voxels` 0, `volume_m3` of the `obb`, the cluster's `free_rays`,
`free_frames_mean`, `observing_views` and `min_camera_distance`, `occ_points` = points,
and `appearance` {`cluster`, `frames_supporting`, `mean_score`, `feat_term`, `geom_term`,
`partners_visible`}).
`instances/<rec>_feats.npy` (N, 1024) float32: L2-normalised DINOv3 ViT-L/16 patch tokens
pooled inside the eroded masks, averaged over views. `instances/<rec>_cls.npy` (N, 1024):
CLS token of the best view's mask box crop. `instances/<rec>_points.npz`: `points` (M, 3)
float32 (1 cm voxels), `colors` (M, 3) uint8, `instance` (M,) int32 id.
`instances/<rec>_masks.npz`: one row per (keyframe, instance) where the instance is
visible (>= `frame_visible_frac` of its points depth-consistent within `frame_depth_tol`,
judged on a fixed 2000-point subsample, or, with `frame_min_points` > 0 (default 0: off),
an estimated `frame_min_points` of them (subsample fraction x point count; 1 cm points,
so about that many cm^2 of the object in view, which keeps an object partly out of the
image; `roomdiff.instances.frame_visible`); SAM prompted with the
box and up to `frame_prompt_points` of those projected instance points, so the masks
follow the grown instance; on the full-resolution keyframe JPEG `keyframes/<rec>/<file>`,
box and points scaled from recon pixels, or on the recon image if the JPEG is missing or
of another aspect ratio): `keyframe` (file name), `instance_id`, `score` (SAM, of that
run), `rle` (COCO RLE counts string) and `size` [H, W] at the recon resolution (the
keyframe mask area-downsampled: set where it covers more than half of a pixel),
`rle_full` (counts at the keyframe resolution, "" where the recon image was prompted) and
`size_full` ([H, W] per row, [0, 0] without `rle_full`); read with
`roomdiff.instances.load_masks` (`full=True`: the keyframe-resolution RLE where there is
one) / `decode_rle`.
`instances/meta.json`: parameters, selected / dropped blobs with the reason (small or
too large unverified, hidden surface, no clear view, no mask, too few points), fragment
and instance counts, `sources` (per recording: geometry / appearance instances),
`appearance` (null without clusters, else `clusters`, `covered` [{cluster, recording,
instance, overlap}], `added` [{cluster, recording, overlap}]), `lift_fallbacks` (per recording: blob ids whose lifting fell back),
`lift_growth` (per blob id: the growth ratio of each prompted view, null without a
connected part), `gap_merges` (per gap-rule merge: `recording`, `kind`, `parts` (blob
ids of both), `near_fraction`, `union_obb_extents`), runtime, peak VRAM. `instances/matching.json`: cosine and cost
matrices of the moved candidates and the accepted pairs (cost = 1 - cosine +
`lambda_size` |log volume ratio|; volumes and the extent-ratio test use the `obb`
extents, which do not change when the world frame turns about +Y), and `replaced`
[[before id, after id, overlap]]: candidate in-place replacements, i.e. a before and an
after appearance instance whose `obb`s, each grown by 2 cm, overlap by more than
`appearance.replace_overlap` (0.3) of the smaller grown volume (greedy one-to-one,
largest overlap first). These pairs skip the moved matching (no 0 m "moves"); every
other appearance instance is matched and typed like a geometric one.

`changes.json`:
```json
{
  "scene": {"before": "...mov", "after": "...mov", "units": "metres (approx.)"},
  "changes": [
    {"id": 1, "type": "moved|added|removed|unverified", "label": "office chair",
     "confidence": 0.86,
     "before": {"instance_id": 3, "centroid": [x,y,z], "aabb": [[x0,y0,z0],[x1,y1,z1]],
                "extents": [dx,dy,dz],
                "obb": {"center": [x,y,z], "extents": [w,h,d], "yaw_deg": 12.5},
                "n_points": 5120, "blob_ids": [9, 14]},
     "after":  {"instance_id": 7, "centroid": [x,y,z], "aabb": [[...],[...]], ...},
     "transform": {"translation_m": 0.92, "rotation_deg": 31.0, "rotation_reliable": true,
                   "R": [[...]], "t": [...],
                   "icp_fitness": 0.82, "icp_rmse": 0.012},
     "evidence": {"free_rays": 412, "observing_views": 9, "appearance_similarity": 0.81,
                  "vlm_verdict": "same object, moved"},
     "description": "The office chair moved about 0.9 m from the desk towards the window.",
     "crops": {"before": "crops/1_before.jpg", "after": "crops/1_after.jpg"}}
  ],
  "summary": "3 changes detected: 1 moved, 1 added, 1 removed; 1 region unverified."
}
```
Every change has all keys; `obb` is the instance's oriented box (`aabb` / `extents` are
kept for compatibility); each side also carries the instance's `source` ("geometry" |
"appearance"). A candidate in-place replacement (`matching.json: replaced`) is written as
a `removed` change with `"replaced_by": <id of the added change>` and an `added` change
with `"replaces": <id of the removed change>` (optional keys, only on such pairs; after M5
the linked change may be in `rejected` or merged away), both with
`evidence.replace_overlap`. Removed / added changes of appearance instances also have
`evidence.appearance` (the instance's appearance terms); `before` / `after` is null on the side where the object does
not exist (removed: after, added: before, unverified: the recording that never observed
it) and `transform` is null unless moved. `R`, `t` map before points to after points
(`x_after = R x_before + t`); `translation_m` = displacement of the before centroid;
`icp_fitness` = symmetric 3 cm fitness (smaller of the two inlier fractions); from
`icp_min_fitness` (0.3) on the ICP motion is used and `rotation_deg` reported, with
`rotation_reliable` true from `icp_reliable_fitness` (0.5) on and false below (the angle
is then shown as "rotation (unreliable)" in the report and viewer and never quoted in a
description); below `icp_min_fitness` `R` = I, `t` = centroid difference,
`rotation_deg` null and `rotation_reliable` false (also for transforms built by M5
re-pairing / merging). Order: moved, removed, added, unverified; by confidence within a type.
Changes of appearance instances have their confidence scaled by the appearance factor A
below (moved: the smaller factor of both sides).
M4 writes `label: "object"`, a templated `description` and `vlm_verdict: null` (M5 fills
them, see below). Evidence keys: removed / added / unverified: `free_rays`, `free_frames_mean`,
`observing_views` (keyframes of the other recording that saw through the place),
`occ_points`, `mask_score`, `view_consistency` (+ `seen_in` for unverified); moved:
`free_rays` (both places), `observing_views` (mean), `observing_views_before_place`,
`observing_views_after_place`, `appearance_similarity` (DINOv3 cosine), `cls_similarity`,
`mask_score`, `icp_fitness`.
Confidence = A x (w_mask S_mask + w_free S_free + w_consistency S_cons) / (sum of weights),
weights `changes.w_mask` / `w_free` / `w_consistency` (0.6 / 0.4 / 0): S_mask = mean SAM
score, S_free = min(observing_views / free_views_full, 1) (moved: mean of both places),
S_cons = view consistency (0.5 if unknown; moved: mean of both); unverified: A x 0.5
(w_mask S_mask + w_consistency S_cons) / (w_mask + w_consistency) <= 0.5. A = 1 for a
geometry instance; for an appearance instance A = 0.5 + 0.5 min(cue / full, 1) in [0.5, 1],
with `changes.appearance_evidence` "score": cue = the cluster's `mean_score` (mean region
score: depth vote share + `appearance.w_feat` x feature dissimilarity), full =
`appearance_score_full` (2.0, its largest value), or "frames": cue = `frames_supporting`,
full = `appearance_frames_full` (4). The earlier formula is w_mask 0.3, w_free 0.4,
w_consistency 0.3 with "frames"; on SceneDiff pairs view consistency did not separate true
from false detections, frames_supporting only on one of two subsets, the SAM score and the
region score on both (README, benchmark section).
Crops (paths relative to `<out>`; JPEG from the full-resolution keyframes, object box
x `crop_context`, at least `crop_min_px` per side): `<id>_<rec>.jpg` shows the object in its
own recording with mask outline and box (keyframe = best by the view scorer on the
instance points, depth-consistent within 5 cm; keyframes with a stored per-keyframe mask
first, ranked by score x SAM score); the other side shows the other recording looking at
the same place (best by the view scorer, points not occluded, at least 15 % visible; box
= the projected points), i.e. the empty place or the new object. Moved objects have both
object crops plus `<id>_before_place.jpg` / `<id>_after_place.jpg` (the other recording at
the old / new place). A side is missing if no keyframe of that recording sees the place.
`changes.rrd` / `scene.rrd`: boxes per change coloured by type (the `obb` when present,
else the AABB), labelled "<id> <type>: <label>", instance points, moved arrows from the
before to the after box centre (`roomdiff/viz_rerun.py`); world view coordinates LUF
(y up) for a levelled reconstruction, else RDF.

## M5 semantics -> `<out>/changes.json` (rewritten), `<out>/semantics/`, `<out>/report.md`
Implemented (`roomdiff/semantics.py`, `roomdiff/vlm.py`, `roomdiff/report.py`, command
`roomdiff describe`; parameters in `configs/default.yaml: semantics`; the prompts are
constants in `roomdiff/semantics.py`). Qwen3-VL-8B-Instruct, bf16, greedy decoding, every
crop resized to <= `image_max_side` px. The first run backs the M4 document up to
`semantics/changes_m4.json`; later runs start from that backup (re-run `roomdiff changes`
to refresh it). Steps: naming -> fragment merging -> re-pairing -> judge -> descriptions
-> summary.

Changes keep the M4 schema and ids (stable, so ids have gaps where entries were rejected
or merged); M5 sets or adds:
* In-place question (`JUDGE_REPLACED`, answer `change` = same | different | removed |
  added | unclear, logged as `judge_replaced:<r>+<a>` / `judge_in_place:<id>`): a linked
  replaced pair (both still removed / added and pointing at each other) is asked once with
  the removed object's crop (image 1) and the added object's crop (image 2): same (only
  lighting / exposure / viewpoint / a slight shift or fold differ) -> both rejected;
  different -> both confirmed; removed / added -> that one confirmed, the other rejected;
  unclear -> both unclear. A single removed / added appearance change is asked with its owner crop
  and the other recording's view of the place (in time order): same -> rejected; its own
  type or different -> confirmed; the other type -> rejected; unclear -> unclear. `evidence.vlm_replace_answer` keeps the answer, `vlm_reason` = "in place:
  <answer>; <reason>". An appearance change whose label names part of the room or a bare
  surface (background nouns, or head noun sheet / bedsheet / mattress / tabletop /
  countertop / worktop / surface / floorboard / tile / plank) is rejected whatever the
  answer (the bed sheet where a hairbrush was later put). Re-pairing never pairs the two
  sides of one replaced pair. Kept pairs get "replaced by a <label>" / "took the place of
  the <label>" descriptions.
* `label`: VLM name, normalised (lowercase, no article, cut at a preposition, <= 3 words);
  `attributes` (colour / material), `object_description` (the VLM's sentence about the
  object).
* `evidence.vlm_verdict` "confirmed" | "unclear" (rejected entries move to `rejected`),
  `evidence.vlm_reason`; unclear: `confidence` x `unclear_factor`. Moved: `motion_kind`
  "rigid" | "deformed" | "unknown" and `evidence.vlm_motion_reason` (`motion_kind` is null
  for the other types). Unverified regions are rejected unless the VLM sees a distinct
  object (not wall / floor / door frame / window / mirror reflection / shadow / lighting).
* `evidence.reason_class` (entries of `rejected` only, `REJECT_CLASSES`): why the candidate
  is not a change. `room_or_surface`: part of the room or a bare surface: an unverified
  region whose label names part of the room (`is_background`), an appearance-channel
  removed / added change whose label names part of the room or a bare surface
  (`surface_rejection`), or a region the object test flags `inside_mirror` /
  `part_of_room`; this class wins even where the judge had rejected the entry anyway.
  `judge_same`: the judge saw no change: an in-place answer "same" or the other type (a
  replaced pair answered "removed" rejects its added side), or the removed / added question
  answered "rejected" (still / already at that place). `judge_rejected`: any other judge
  rejection: the object test found no distinct object (unverified regions, removed / added
  changes without a view of the place), or a half of a split moved change (below) was
  rejected. An entry keeps a class it already has: M4 sets `evidence.reason_class` on
  unverified changes, `never_observed` (the other recording never looked at the place) or
  `weak_alignment` (a geometry candidate demoted because the two reconstructions disagree),
  so a rejected unverified region shows its M4 class. Changes the optional SAM 3 re-check
  moves to `rejected` get no class (`evidence.recheck_verdict` says why).
* A `moved` change (M4 geometric match or re-pairing) whose match the judge rejects is not
  dropped: it is split into a `removed` change (its `before` side; id = the moved id, or
  the first merged candidate's) and an `added` change (its `after` side; id = the second
  merged candidate's, else the next unused id), each named again from its own crop and
  judged like any removed / added change (the removed / added prompt with the object
  crop and the other recording's view of the place; the object test without one):
  "confirmed" keeps the moved confidence, "unclear" gives confidence x `split_factor`
  (0.6), "rejected" (e.g. the half is a wall region) moves the half to `rejected`;
  `vlm_verdict` / `vlm_reason` are the half's own, `evidence.split_from` = the moved id,
  `evidence.split_reason` "appearance judge did not confirm the match: <judge reason>",
  `transform` / `motion_kind` null, crops: removed `before` / `after` = moved `before` /
  `before_place`, added `after` / `before` = moved `after` / `after_place`; the other
  evidence keys come from the instance record (as M4 writes them for removed / added) or,
  without it, from the moved per-place evidence (`observing_views`, `mask_score`). The
  kept halves then go through fragment merging and re-pairing once more (the steps
  below), only in pairs with another change, never with each other: a wrong M4 match can
  so become the right one (office_1: wrong chair pair split, the small chair's halves
  re-paired, the large chair's upper half joined with its separately found base).
* Fragments of one object (`enable_fragment_merge`): two `removed` (or two `added`)
  changes with matching labels whose oriented boxes are at most `fragment_gap_m` (0.15 m)
  apart (exact gap of the two vertical prisms) are shown to the VLM with both owner crops
  ("two parts of one physical object, e.g. the seat and the base of one chair, rather than
  two separate objects?"); closest pair first, each change in at most one merge. If
  confirmed they become one change: the fragment with more points keeps its id, type,
  label (replaced by the VLM's name of the whole object if that matches one of the two
  labels), attributes, crops (+ `crops.<side>_part` = the other's owner crop) and evidence,
  with `free_rays` / `occ_points` summed, `observing_views` = max, `merged_from` [kept,
  absorbed], `merged_types`, `merged_part_reason`, `fragment_gap_m`; the side becomes the
  union: centroid, `aabb`, `extents` and `obb` of the union of both instances' points
  (`instance_ids` [both], `n_points` summed, `blob_ids` united; box union without the
  points); confidence = max; the description is regenerated (step below). Then judged
  like any other change. The masks of all `instance_ids` are united per frame for the
  SceneDiff evaluation.
* Re-paired removed R + added or after-only unverified A (matching labels: same label,
  head noun or synonym group; centroids < `repair_max_distance`; VLM: same object) -> one
  `moved` change with R's id: `before` = R's side, `after` = A's, `transform` from the
  centroids (`R` = I, `rotation_deg` / `icp_*` null), confidence = mean, `crops`
  `before` = R's, `after` = A's, `before_place` = R's after view, `after_place` = A's
  before view, `evidence.merged_from` [R, A] and `merged_types`.
* A moved change M absorbing an added change P (matching labels, gap between the after
  boxes (oriented boxes if present) <= `merge_max_distance`, VLM: parts of one object, e.g. a spread-out quilt):
  `after` = union box (`instance_ids` [M's, P's], `n_points` summed, point-weighted
  centroid, `obb` = oriented box around the corners of both `obb`s), transform =
  centroid translation (rotation null), confidence = mean, `evidence.merged_from` [M, P],
  `evidence.merged_part_reason`, `crops.after_part`.
* `description`: one sentence (label, type, distance in m with one decimal, size (`obb`
  extents if present, else the AABB's), coarse place "far left part of the room" along
  the mean camera direction, height of the lowest
  observed part above the floor = median of the lowest 5 % of all valid points, evidence
  counts), polished by one text-only VLM call if every number and the label survived;
  `description_template` keeps the template sentence.

Top level: `rejected` (same schema, `vlm_verdict` "rejected" + reason + `reason_class`),
`summary` = "N changes: a moved, b added, c removed; d unverified." + a VLM paragraph.
`semantics/meta.json`: model, params, runtime per step, VLM calls and seconds, counts in /
out, rejected (id, type, label, reason), merges (every VLM pair check: kind, ids,
labels, distance or box gap, verdict, reason; fragment checks also the VLM's `label`),
splits (rejected moved changes: `id`, `label`,
`into` [removed id, added id], `labels`, `reason`, `verdicts` of the two halves), verdict
per id, `replacements` (per in-place pair question: `ids`, `labels`, `answer`, `reason`),
polished /
template-kept counts, scene frame (up vector: +Y in a levelled frame, else the camera
estimate; floor height), `peak_vram_gb`, `peak_ram_gb`.
`semantics/log.jsonl`: one line per VLM call `{tag, images, prompt, answer, seconds}`.
`report.md`: title, summary, counts table, one section per change (type badge, label,
confidence, verdict, motion, description, evidence line, crops as relative markdown
images), collapsible rejected list with the reason class as a tag and the VLM reasons,
coverage note from `recon/meta.json`.

## Re-check (optional) -> `<out>/changes.json` (rewritten), `<out>/recheck/`, `<out>/report.md`
Implemented (`roomdiff/recheck.py`, SAM 3 wrapper `Sam3TextSegmenter` in
`roomdiff/models.py`, command `roomdiff recheck`; parameters in `configs/default.yaml:
recheck`; `enabled: false` by default: nothing is read or written, and `roomdiff run` runs
the stage after `describe` only when it is true). SAM 3 (`third_party/sam3`, the official
`facebook/sam3` `sam3.pt`; built on the GPU, then the checkpoint memory-mapped into it, bf16
autocast) is prompted with each change's label on the full-resolution keyframes; its
masks (detections from score 0.1) are compared at the reconstruction resolution. The first
run backs the M5 document up to `recheck/changes_m5.json`; later runs start from that
backup (re-run `roomdiff describe` to refresh it). In one `roomdiff run` process the stage
refuses to load SAM 3 while more than 4 GiB are still allocated on the GPU (Qwen3-VL not
freed), and frees SAM 3 when done.

Every kept `removed`, `added` and `moved` change (VLM verdict confirmed or unclear) is
checked; unverified changes are skipped and the M5 `rejected` list is left alone:
1. Prompt = `label`; a label naming only a material or a vague thing (head noun fabric,
   cloth, leather, wood, plastic, metal, paper, cardboard, glass, item, thing, object, ...)
   falls back to the head noun of `object_description` (last word of its first noun
   phrase: "A crumpled black jacket lying on the floor." -> "jacket"); none -> skipped.
2. Own side (moved: both recordings): the `max_views` (2) best keyframes of the object's
   recording with a stored per-keyframe mask (`instances/<rec>_masks.npz`, united over
   `instance_ids`), ranked by view score x SAM score as the crops, viewing directions
   `instances.min_view_angle` apart when possible (without stored masks: the best views
   by the view scorer, reference = the projected instance points). Found = a detection
   with score >= `score_min` (0.5) and mask IoU >= `overlap_min` (0.3) with the reference.
3. Places in the other recording: removed -> after views of the old place, added -> before
   views of the new place, moved -> both (old place in the after recording, new place in
   the before recording). Views = the crop place views (`select_views`, not occluded, >=
   15 % of the points visible), `max_views` per place. Place region = the instance points
   projected into the view that are not hidden behind a nearer surface (observed depth >=
   z - 5 cm), gaps between the 1 cm points closed by a morphological closing (square of
   2 ceil(f 0.01 / z) + 1 px, 3..15), with the nearest point depth per pixel; moved:
   minus the pixels of the object's other position (dilated by 2 px), a view keeping less
   than 25 % of the place is replaced by the next one. A detection is at the place if
   score >= `score_min`, IoU with the region >= `overlap_min` and >= 50 % of its pixels
   inside the region (valid depth) lie within `depth_tol` (0.15 m) of the object's depth
   (not an instance behind or in front of the place).
4. Verdict: `skipped` (no prompt, no view or not checked), `contradicted` (>= 1 view with
   the prompt at the place; moved: at both places, since one place alone is often a
   sibling such as the other curtain of a window), `unsupported` (not found on the
   object), else `supported`. Confidence x `unsupported_factor` (0.9) without own-side
   support and x `contradicted_factor` (0.6) when contradicted (rounded to 3 decimals). A
   change moves to `rejected` only when contradicted in >= `reject_views` (2) views: its
   `vlm_verdict` / `vlm_reason` stay, `evidence.recheck_verdict` = "contradicted"; the
   summary gets the new counts and a sentence naming the rejected changes.

Changes keep the M5 schema, ids and order rules; every kept change (and every change the
re-check rejected) gets `evidence.recheck`:
```json
{"prompt": "black jacket",
 "own": {"views": [{"recording": "before", "frame": 43, "file": "before_00043.jpg",
                    "detections": 2, "score": 0.84, "overlap": 0.986, "found": true}],
         "found": true, "score": 0.84, "overlap": 0.986},
 "other": {"views": [{"place": "old", "recording": "after", "frame": 52, "file": "...",
                      "detections": 0, "score": null, "overlap": null, "at_place": null,
                      "found": false}],
           "found": false, "found_views": 0, "score": null, "overlap": null},
 "verdict": "supported", "factor": 1.0,
 "reason": "'black jacket' found on the object in 2 of 2 own views (0.84), absent from the old place in 2 after views"}
```
Per view: the best detection (found first, then by IoU and score; `score` / `overlap` /
`at_place` null without detections), `reference` "points" on own views without a stored
mask. `own.found`: found in every recording with own views (null without own views);
`other.found`: the prompt is at a place in some view (null without place views),
`found_views` = such views; skipped changes have `own` / `other` null.
`recheck/log.jsonl`: one line per keyframe and prompt `{tag (role:id:recording|place, all
checks using it), images [keyframe path below <out>], prompt, answer [{score, box [x0, y0,
x1, y1] at recon resolution, area}], seconds}`. `recheck/meta.json`: model, checkpoint,
`missing_keys`, params, runtime per step (load, plan, load_model, inference, decide,
total), images / prompts, `seconds_per_image` {mean, median, max}, `counts` per verdict +
rejected, `verdicts` and `confidence` [before, after] per id, `rejected` (id, type, label,
reason), `peak_vram_gb`, `peak_ram_gb`. The report's evidence line adds "SAM 3 re-check
<verdict>: <reason>" (not for skipped changes); its rejected list shows the re-check reason
for changes the re-check rejected.

## M6 outputs
`scene.rrd` (Rerun; written by `roomdiff viz`: reconstruction, M3 candidate blobs,
appearance score maps (`world/<rec>/cam/<i>/appearance`: the keyframe dimmed with a
black-red-yellow-white heat map of the score, for the cameras logged with their image)
and clusters (`world/appearance/<rec>/<id>`, oriented boxes, "covered" if an M4 instance
already held them), changes), `viewer/` (`roomdiff viewer`: static web viewer assets: the
app (index.html, app.js, style.css and three.js r160 vendored in `vendor/three/`, so it needs
no network), downsampled clouds, boxes, crops, changes.json; `scene.json: up_axis` = [0, 1, 0]
for a levelled
reconstruction, else the camera-based estimate; `view_dir` = the horizontal direction of
the viewer's home view, chosen among 12 azimuths as the one whose home camera sees the
largest unoccluded share of the points, `viewer_export.best_view_dir`). The viewer draws each change's `obb`
when present (else its `aabb`) and the moved arrow between the box centres.

## Evaluation (SceneDiff) -> `<out-root>/`
Implemented (`roomdiff/eval_scenediff.py`, `roomdiff eval`, parameters in
`configs/default.yaml: eval`). Scoring is done by the official, unmodified
`third_party/scene_diff/scripts/evaluate_multiview.py` (MIT).

Layout: `runs/<pair>/` (one `roomdiff run` per pair with `ingest.mode=fixed`, GPU-locked
through `scripts/run_guarded.sh -g`; pairs that already have `changes.json` are skipped),
`logs/<pair>.log`, `pred/<pair>/object_masks.pkl` + `prediction.json` (objects per type,
masks per video, mapped benchmark frames, mask resolution, mask / GT / target sizes,
aspect check),
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
`eval.include_unverified` is true (default false: they claim no change; when submitted,
their confidence is <= 0.5, so they rank after the verified changes and could only add
recall to AP, which they never did, while their pixels lower the global IoU). `cost` = change confidence x SAM mask score (the evaluator ranks per-view
detections by it and objects by its mean). Changes without a mask at any mapped frame are
left out. Masks are COCO RLE (upright, same aspect as the GT; checked in
`prediction.json: aspect_ok`) at the keyframe resolution with `eval.mask_resolution: full`
(`rle_full` of `<rec>_masks.npz`, the recon-resolution mask where a row has none; e.g.
1024 x 576 for 1920 x 1080 portrait GT) or at the reconstruction resolution with `recon`
(the default, e.g. 672 x 378; the two measured identical to three decimals); the
evaluator resizes them with nearest neighbour. A failed run
gets an
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
