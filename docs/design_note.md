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
blob. Those objects need an appearance cue, which we discuss under limitations.

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

## Limitations we know about
- Small, flat or thin objects are missed by the geometric channel (recall, not precision).
- Deformable changes (a curtain opened, a quilt spread out) get a "moved" verdict with an
  unreliable rotation; the judge marks them as deformed rather than rigid.
- Mirrors create unverified regions behind the glass; a mirror mask would fix it.
- Scale is approximately metric; distances are reported with that caveat.
- Each run takes about 1.5–2.5 minutes on an RTX 5090, most of it model loading.
