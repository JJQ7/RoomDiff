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
"same" answer deletes a change found with confidence 0.9. (2) Pi3X's metric scale is not
to be trusted on close-range, top-down video: `scripts/lidar_check.py` fits our clouds to
the LiDAR meshes with 1.7-2.9 cm median residual, but at a scale of 0.55-0.57 for all six
recordings (the two scans of a pair agree within 0.7-3.6 %), i.e. 1.8x too large, while
the relative geometry (a 0.44 m move, a 54 deg turn) is fine. (3) One Pi3X pass needs one
image size: mixed portrait / landscape pairs are handled by centre-cropping both videos to
squares (see below), with a warning from `scripts/check_video.py`.

## What we tried and did not keep as default
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
