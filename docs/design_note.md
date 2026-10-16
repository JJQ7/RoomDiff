# Design note: choices and trade-offs

## The problem as we framed it
Two casual phone videos of the same room, taken from different paths, differ for three
reasons: objects really changed, the cameras looked at different parts of the room, and
objects hid other objects. A change detector is only useful to a robot if it separates the
first reason from the other two. We therefore treat every candidate change as a hypothesis
and demand evidence from the *other* recording before we call it a change.

## 1. Shared 3D frame without a registration step
Both videos' keyframes (about 48 each, picked for sharpness) go through one forward pass of
Pi3X, a permutation-equivariant feed-forward reconstruction model. Because the model does
not privilege any reference view, the cameras of both recordings come out in one metric
frame directly; there is no SfM, no ICP alignment and nothing to tune. On the SceneDiff
benchmark scenes we measured a median distance of 0.5–0.9 cm between the two recordings'
surfaces in the co-visible area, and point-to-plane ICP could not improve it. A separate
mode (one pass per recording plus a Sim(3) anchor alignment) exists as a fallback and was
consistently worse, mainly because of a relative scale error between passes.

Trade-off: feed-forward models give approximate metric scale (we estimate ±5–8 %) and lose
thin structures; COLMAP-style SfM would be more precise on texture but is slow and brittle
on scenes where objects moved. Alignment-free joint inference also means both videos must
be processed together; it does not scale to many revisits without a map.

## 2. Visibility reasoning: "seen empty" is not "not seen"
Each recording gets a three-state voxel grid (occupied / free / unknown) at about 2 cm.
Free space comes from carving the rays between every camera and the surface it observed;
unknown is everything never crossed by a ray. A before-voxel is only *disappeared* if the
after recording carved it as free; if the after recording never looked there it stays
*unverified*. A distance-transform margin (4 cm) around the other recording's surfaces
absorbs residual misalignment and depth noise, and a small morphological opening removes
the thin shells that remain.

On three benchmark scenes every disappeared/appeared blob corresponded to a real change and
there were no false positives. The price is recall on small or flat objects: a wallet on a
table or a bag lying flat is within the margin of the surface it rests on and produces no
blob. Those objects need an appearance cue, which is the second channel below.

*Alignment gate.* The margin assumes that the two reconstructions agree to a centimetre or
two. The recon stage checks this on the co-visible points, in both directions: WEAK if the
median nearest-neighbour distance exceeds 2 cm, fewer than 90 % of the points lie within
10 cm, or the recordings barely overlap; FAIR if only the share within 3 cm drops below
80 %; else OK. FAIR is a warning and changes nothing: it catches the tabletop runs with one
recording turned a quarter turn (74 % within 3 cm, the whole table top reported "added"),
which passed the other two tests, but also three benchmark pairs without such errors
(77-79 %). With `visibility.weak_alignment_demotes` (off by default, see below) WEAK makes
the voxel channel report its removed / added blobs as unverified, since a misaligned
surface looks "seen empty" from the other side. The appearance channel is left alone: its candidates come from keyframe
pairs whose depths agree after reprojection. One pair in each 20-pair subset is WEAK. On
gas_station_3_gas_station_4 (median 7-9 cm, 28 % within 3 cm) all 7 benchmark predictions
were voxel changes, 6 false (signs, a bumper, a gas pump "added") and 1 right (a car driven
away); the gate removes all 7 (reported pairs: FP 39 -> 33, TP 36 -> 35, per-scene AP 0.413
-> 0.400, typed 0.280 -> 0.264). On street_3_street_4 of the development subset (median
3-4 cm, 40 % within 3 cm) all four voxel changes were right (a van removed, three cars
added); the van and two cars, whose demoted blobs exceed the 1.5 m size limit of
unverified regions, came back through the appearance channel, the third car stayed
unverified (TP 25 -> 24, per-scene AP 0.459 -> 0.436). The gate trades a few correct
changes for fewer false ones and lowers every AP metric on both subsets, so it is off by
default; the verdict and the 3 cm share stay in the report as a diagnostic, and the WEAK
pairs of this benchmark are outdoor scenes whose true changes are cars, exactly what the
voxel channel finds there. A size rule (demote only blobs shorter than about 1.2 m, which
separates the cars from the signs and bumpers on these two pairs) is fitted on 11 changes
and was not adopted.

*Floor is not an object.* An unverified blob is occupied space the other recording never
looked at, and the floor is occupied space: on the top-down tabletop videos the first
run's unverified regions were patches of floor that one video had filmed, joined through
the floor with whatever stood on them, and SAM's mask then followed the floor and the
table edge to 3.4-3.7 m; the judge, shown a crop with a stool or a suitcase in it,
confirmed "a distinct object" and named it so. Since 16 October the voxels less than 6 cm
above the fitted floor plane are left out of the two unverified masks before the
components are labelled (`visibility.unverified_floor_cut`, only when the recon levelled
the frame on a floor plane; the disappeared / appeared masks are untouched, since a
removed object leaves free space, not floor). Objects standing on an unobserved patch of
floor keep their own blobs: tabletop_1's four strips become the suitcase and the chair,
tabletop_3's 2.6 m "wooden table" the stool, bucket, brush and cookie box on that floor,
and every reported change of the three pairs stays the same. On the 20 reported benchmark
pairs the unverified regions that pass the size filters go from 109 to 111 (some floor
patches vanish, others split into the objects on them); they are not scored. A fresh
end-to-end run of those pairs with the cut differs from the previous one in a single pair,
where a removed cardboard box is now re-paired with an unverified region of the after video
that the floor no longer inflates and comes out as moved: objects 37 / 34 / 38 -> 38 / 33 /
37, per-view / per-scene / typed AP 0.468 / 0.433 / 0.316 -> 0.474 / 0.444 / 0.325.

## 2b. Appearance channel: what the voxels cannot see
For every keyframe we pick the two best co-visible keyframes of the other recording (same
surfaces, depth-consistent after reprojection). Each DINOv3 patch of a keyframe is
reprojected into its partners; where the partner looks at the same place we compare the
two patch features (1 - cosine), and where the partner sees *through* the place (its depth
is clearly larger than ours) we add a depth vote. The per-keyframe score map is thresholded,
each connected region prompts SAM 2.1, the masks are lifted to 3D and clustered across
keyframes, and clusters that a geometric instance already covers are dropped. The rest
become instances tagged `source: appearance` and flow through the same matching, judge and
description stages; a before/after pair of appearance instances in the same place is first
asked a single in-place question (same thing, replaced, removed or added), which is how a
jumper on a bed or a bottle swapped for another gets a verdict without a fake 0 m "move".

An appearance change stands on weaker evidence than a voxel change: no free-space test is
applied to it. It rests on keyframes of the other recording that looked at the same place
(`observing_views`) and saw different features there, the see-through depth vote where there
is one (`evidence.appearance.geom_term`), and the judge's in-place answer on the paired crops.

On the same three scenes this recovered the wallet, the hairbrush and the tissue box, at
the cost of one false positive (a bare wall patch that the judge read as a "wall
decoration", gone once the prompt said the drawn outline is not part of the room); on 20
SceneDiff validation pairs it lifted per-view AP from 0.15 to 0.40 (see the README
benchmark table). Denser keyframes (every 0.5 s instead of 1 s) give every keyframe more
co-visible partners and raise the per-scene metrics a little more. The judge is the weak
point of this channel: a lighting-only region survives if the VLM over-reads the crop, and
an object that moved a few centimetres may be judged "same".

## 3. Objects, not voxels
Blobs fragment (a removed chair became four blobs) and merge neighbours. We therefore use
blobs only as *seeds*: each blob is projected into its best views, SAM 2.1 is prompted with
the box and a few visible points, and the mask is lifted back to 3D and bounded by the
blob's neighbourhood so floor and wall are not swept in. Masks that overlap across views
are merged into one object, and touching fragments of the same kind (a chair seat and its
base, 5 cm apart) are joined if the union stays object-sized. DINOv3 patch features pooled inside the masks give each object
an appearance vector; Hungarian matching between disappeared and appeared objects, gated by
feature similarity and size, turns a pair into a *moved* object with an ICP-estimated
rigid transform (distance in metres, rotation in degrees when the fit is reliable).

## 4. Semantics as a judge, not a decoration
A local vision-language model (Qwen3-VL-8B) names every object from its crops and writes
the description, but its more important job is adjudication: for a *removed* object it
sees the object and the same place in the after video and must agree the object is gone;
for a *moved* pair it must agree the two crops show one physical object; for *unverified*
regions it must say whether there is a distinct object at all (walls, door frames and
mirror reflections are rejected). Same-label objects that geometry split into "removed
here" and "appeared there" are re-paired into one *moved* object, and same-label removed
(or added) fragments a few centimetres apart are merged when the model sees parts of one
object. No API keys are used;
everything runs offline on one GPU.

*Re-pairing by appearance.* Of the 29 moved objects of the reported pairs, 5 came out as
moved, 7 as a removed or an added half only and 17 not at all. We traced each of these 24
to our instances (mask IoU with the ground truth per video, then the DINOv3 cosine, sizes
and distance of the two halves and the step that kept them apart). 17 are lost upstream:
one side was never segmented as an object of its own. Two are correct M4 pairs that the
judge split (a soy sauce bottle, a cream bottle), two have masks too partial to reach IoU
0.5, and three were kept apart although both halves were found. The M4 cosine threshold
stopped none of those three (another cream bottle failed the M4 size ratio, 2.04 against
2.0, because its before instance also holds a jar next to it); in M5 the names did, since
re-pairing proposed only pairs with matching labels: "stainless steel saucepan" and
"stainless steel pot", "plastic bottle" and "bottle of lotion", "wooden ladder" and
"wooden ladder shelf" (the ladder moved 0.2 m, so its two sides were also an in-place
pair, which re-pairing never joined). Re-pairing now also proposes a removed + added pair
whatever its names, an in-place pair included, when the best-matching instances of the two
have a DINOv3 cosine of at least 0.85 and pass the M4 size gate; the strict same-object
judge still decides. On the 40 pairs of both subsets this asks five more questions: the
ladder (cosine 0.91) and a letter in the mailroom pair (0.95) are confirmed, the pot
(0.94; the judge sees different handles), a second letter and two different signs are not,
and that cream bottle (0.70) is not proposed. That is one more moved object found as moved
on each 20-pair subset, with one or two false positives fewer (README, benchmark).
Lowering the M4 threshold instead (0.60, 0.55 or 0.50 rather than 0.65), replayed on the
stored candidates of both subsets, finds no further correct pair, adds one to three wrong
ones and loses a correct one.

*Describe before deciding.* The judge's commonest mistake was to answer "same" or
"rejected" for an object that is plainly in one image only (a biscuit box found by the
appearance channel with confidence 0.9 was deleted this way). The removed / added and
in-place prompts now ask for the content of each box first, in the JSON answer, before
the verdict; the in-place judge also says whether that content is a movable object or a
bare surface, which turns "different: red box vs. shelf" into "removed" rather than a
removal plus a spurious "added shelf". On a disjoint 20-pair subset of SceneDiff val this
raised per-scene AP from 0.40 to 0.46; on the reported 20 pairs from 0.38 to 0.41 (typed
AP 0.26 -> 0.28, 4 more objects found).

*Confidence from the evidence that separates.* The benchmark ranks every detection by our
confidence times its SAM score, and that confidence barely told true from false detections
apart (reported pairs: mean 0.86 vs 0.80). We labelled every submitted object with the
evaluator's own matching and ranked the objects by one evidence number at a time (AUC on
the development / reported pairs, 0.5 = chance): the SAM mask score separates them well
(0.87 / 0.83) and the appearance channel's mean region score moderately (0.68 / 0.66),
while the view consistency of the mask features, 0.3 of the old weights, does not (0.44 /
0.50), and the keyframes supporting an appearance region, the old appearance factor, only
on one subset (0.65 / 0.48). The default now gives the consistency weight to the mask
score (0.6 mask, 0.4 free space) and scales a change that only the appearance channel
found by 0.5 + 0.25 x its region score (at most 1), so it ranks below a geometric change
with equal mask and free-space evidence unless its cue is strong. Re-ranking the same
detections, per-view / per-scene / typed AP went from 0.478 / 0.459 / 0.401 to 0.487 /
0.488 / 0.408 on the development pairs and from 0.432 / 0.413 / 0.280 to 0.442 / 0.416 /
0.296 on the reported ones. Three alternatives were measured and not kept: the see-through
share alone as the appearance factor and a free-space term from the free count per voxel
did well on the development pairs (per-scene AP 0.486 and 0.476) but lost on the reported
ones (0.388 and 0.402), because flat objects such as a closed laptop on a bed leave
neither a depth cue nor free space; and a view term that does not saturate at 5 views
(log(1 + views) / log 21) lowered the APs on both subsets (reported per-scene AP 0.387),
since appearance changes are seen from many views whether they are right or wrong.

## 5. Outputs
`changes.json` (typed changes with 3D boxes, transforms, confidence and the evidence behind
each verdict), `report.md`, a Rerun session for inspection, and a self-contained web
viewer (three.js) that shows both clouds in the shared frame, the change boxes, the
natural-language report and the before/after crops, and runs on GitHub Pages.

## 6. A held-out check on our own recordings
Three pairs of iPhone videos around a low table (consecutive states; the staged changes
were written down first, `examples/tabletop_*/gt.yaml`), each run once with the default
settings. Of 9 staged changes, 5 came back with the right type and object, 2 as a change
of one type only (a bucket's new place found but not its old one; a tea canister whose
old place was seen empty while its new place, under the table edge, was never observed),
2 were missed (a biscuit box the VLM judge wrongly called present in both crops; a stool
pushed under the table that was never segmented), with 1 false positive (a bin in the
background "moved" 0.10 m). Three lessons. (1) The judge is the weakest link: a single wrong
"same" answer deletes a change found with confidence 0.9; the "describe before deciding"
prompt of section 4 recovers that box when the describe stage is re-run on the same
reconstructions, but on these close-range videos it also accepts four table and chair
regions seen from different viewpoints, so here it trades precision for recall. (2) Pi3X's metric scale is not
to be trusted on close-range, top-down video: `scripts/lidar_check.py` fits our clouds to
the LiDAR meshes with 1.7-2.9 cm median residual, but at a scale of 0.55-0.57 for all six
recordings (the two scans of a pair agree within 0.7-3.6 %), i.e. 1.8x too large, while
the relative geometry (a 0.44 m move, a 54 deg turn) is fine. (3) One Pi3X pass needs one
image size: mixed portrait / landscape pairs are handled by centre-cropping both videos to
squares (see below), with a warning from `scripts/check_video.py`. (4) Floor is not an
object: the first run's unverified regions on these top-down videos were floor patches
named after whatever stood on them (section 2, floor cut).

*No-change control (14 October).* The before video of tabletop_2 cut into its first and
second half, two sweeps of the same unchanged table, run as a pair: the reconstructions
align (91 / 84 % of the co-visible points within 3 cm), the voxel channel finds no
disappeared and one appeared blob, yet four changes survive the judge, all table and chair
parts from the appearance channel (a "moved" table, a "removed" chair leg and table, an
"added" table corner), six candidates are rejected. The same run with the final code of
that day gives the same four. On close-range, top-down video the false positives are the
appearance channel's and the judge's, not the geometry's; see the README held-out section
for the re-run of the three staged pairs with that code.

## What we tried and did not keep as default
*Masks at the keyframe resolution.* The benchmark evaluator matches masks at IoU 0.5 after
resizing them to 1024 px, and ours come out of SAM at the reconstruction resolution
(672 x 378), so coarse boundaries looked like a plausible loss. Prompting SAM with the
1024 px keyframe instead (same internal size, no extra cost) and submitting those masks
changed every metric by less than 0.001 on both 20-pair subsets. The option stays
(`eval.mask_resolution`), the default is the cheaper one.

*Masks in more keyframes.* The evaluator matches objects by the mask IoU stacked over all
benchmark frames of both videos, so a frame where only the ground truth has a mask counts
fully against us. On the reported pairs, 11 near misses (a prediction and a ground-truth
object at stacked IoU 0.1-0.5) are above 0.5 on the frames both have. Of their 57 missing
frames, 41 are in the other video: all 11 ground-truth objects moved, and for 9 of them we
reported only a removal or an addition (for five, the other half is a separate change of
ours, so these are pairing misses: a lotion bottle in two pairs, a pot, a soy-sauce
bottle, a ladder). 10 are views of a quilt or a curtain beyond the part our instance
covers, and 6 show the object partly out of the image, with 5-14 % of its points in view,
below the 20 % of `instances.frame_visible_frac` (the points in view are depth-consistent,
so a larger `frame_depth_tol` does not help). `instances.frame_min_points` also accepts a
keyframe in which an estimated N of the object's 1 cm points are consistent, about N cm²
in view. With N = 200, regenerating only the masks of the stored runs filled 23 of the 30
partly visible ground-truth frames of all submitted objects on the reported pairs (3 of
10 on the development pairs). On the development subset per-view AP rose from 0.478 to
0.490 and pixel IoU from 0.442 to 0.452, with per-scene and typed AP and the 25 / 11 / 23
objects unchanged. On the reported pairs per-view AP rose from 0.430 to 0.462 and IoU
from 0.471 to 0.490 with the same 36 / 39 / 39 objects, but per-scene AP went from 0.406
to 0.405 and typed AP from 0.277 to 0.275, through the ranking, which uses an object's
mean frame score: a false positive made of two fragments now has the masks of its
better-scoring part in three more frames and passed six true positives (0.60 -> 0.72),
and two true positives whose IoU rose (0.90 -> 0.97, 0.85 -> 0.97) scored a little lower,
partial views having lower SAM scores. Those references are the old rule with the masks
prompted at the keyframe resolution, as the code now does; the published row's masks
were prompted on the reconstruction image, and that alone moves per-scene AP from 0.413
to 0.406 with identical matches. A fraction of 0.10 or 0.05, or N = 100, gave per-view AP
0.488, 0.493 and 0.501 on the development subset but cost 0.002 of per-scene and typed AP
there. N = 200 is the default: per-view AP and pixel IoU gain on both subsets with the
same objects matched, and the per-scene / typed changes of 0.001-0.002 are noise at this
sample size (the option can be switched off with `instances.frame_min_points: 0`).

*A bigger appearance backbone.* SceneDiff's own method uses DINOv3 ViT-H+/16, so we swapped
it in for ViT-L/16 (`instances.dino_model`, both stages). On the 20-pair development
subset per-view AP rose from 0.478 to 0.510, but per-scene AP fell from 0.459 to 0.373 and
typed AP from 0.401 to 0.297, with twice the false positives (24 vs 11): the appearance
channel's automatic threshold and clustering were shaped around ViT-L's score distribution,
and the larger model is not a drop-in gain. Kept as an option.

*A bigger judge.* Qwen3-VL-32B-Instruct in place of the 8B model, loaded with its
language model in nf4 (`semantics.quantize_4bit`, 19.5 GiB; the pre-quantized Hub
checkpoints do not load on transformers 5), same prompts, same reconstructions. On the
development subset it answered the biscuit-box question right but did not help overall:
per-view AP 0.481 vs 0.478, per-scene 0.396 vs 0.459, typed 0.342 vs 0.401, 23 / 12 / 25
objects vs 25 / 11 / 23; it splits a few moved objects into removed + added pairs and
drops a few removals the 8B kept. On the reported 20 pairs, run afterwards for
completeness, it came out ahead instead (per-view AP 0.448 vs 0.432, per-scene 0.439 vs
0.413, typed 0.285 vs 0.280, 38 / 35 / 37 vs 36 / 39 / 39). A change that helps one
20-pair subset and hurts the other is noise at this sample size, and the development
subset decides, so the 8B stays: the judge prompts and factors were developed with its
answers, and a different model needs its own round of that. Kept as an option.

*Submitting unverified regions.* They were submitted ranked last so that the AP metrics
could use them; none of them matched a ground-truth object, every AP stayed identical
without them, and the merged-mask pixel IoU rose from 0.37 to 0.47. They are no longer
submitted (`eval.include_unverified`).

*Turning one recording a quarter turn for mixed orientation.* Our first fix for a portrait
+ landscape pair rotated the after video's keyframes 90 degrees so that both sets had one
shape without losing pixels. The joint reconstruction then aligned the two recordings less
well (points within 3 cm of the other recording: 74-80 % instead of 85-96 %) and the
turned recording's table top sat a few centimetres off, so the voxel channel reported the
whole table as "added" and the VLM, shown sideways crops, confirmed it (7 and 4 false
positives on the two pairs). Centre-cropping both videos to squares costs 44 % of each
frame but keeps every image upright: 0 false positives on the same pairs.

*SAM 3 re-check.* After naming, SAM 3 is asked for the change's label (text prompt) on the
object itself and at its place in the other video; a label found at a "now empty" place
would contradict a removal, a label found at the new place supports a move. On the three
reference scenes every correct change was supported and nothing was contradicted; on the 20
benchmark pairs no contradiction fired either, and a small confidence penalty for labels
SAM 3 could not find on the object lowered per-scene AP from 0.383 to 0.374, because SAM 3
is sensitive to wording ("closed laptop" scores 0.06 where "laptop" scores 0.92). The stage
stays in the code as optional evidence (`recheck.enabled`), without the penalty.

## Limitations we know about
- Small, flat or thin objects are missed by the geometric channel; the appearance channel
  recovers many of them but depends on co-visible keyframes (at about 1 keyframe per
  second a scene can have none) and on the VLM judge.
- Deformable changes (a curtain opened, a quilt spread out) get a "moved" verdict with an
  unreliable rotation; the judge marks them as deformed rather than rigid.
- Mirrors create unverified regions behind the glass; a mirror mask would fix it.
- Scale is approximately metric: 1.8x too large on our close-range tabletop videos (LiDAR
  check), so distances are reported with that caveat and converted when a scan exists.
- Each run takes about 2–2.5 minutes on an RTX 5090, a third of it model loading.
- Low overlap between the two recordings is the failure mode: on the 55 `kitchen` pairs of
  the SceneDiff validation set (short sweeps, cluttered counters, a median 42 % of the
  points co-visible against 72 % on `varied`, Pi3X confidence 0.46 against 0.71) the
  pipeline finds 40 of 160 objects and no moved object typed right (per-scene AP 0.18
  against 0.52 on `varied`), because both channels need the other recording to have seen
  the place: the voxel channel through a trustworthy alignment, the appearance channel
  through co-visible keyframes (README, benchmark section).
- On a no-change pair the appearance channel plus the judge still report changes: four on
  our control (the two halves of one tabletop recording, section 6), one false added object
  on the benchmark's own no-change pair (living_room_29_living_room_30).
