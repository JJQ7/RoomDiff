# Capture protocol (iPhone 13 Pro)

## Phone settings (once)
* Settings > Camera > Formats: **Most Compatible** (H.264, avoids HEVC decoding issues).
* Settings > Camera > Record Video: **1080p at 30 fps**, **HDR Video off** (10-bit Dolby
  Vision complicates decoding and changes colours between takes).
* Lock screen rotation; hold the phone **landscape** for both recordings of a scene.
* Clean the lens. Keep the same lighting for both recordings (same lamps, same curtains).

## Scene set-up
* Living room or bedroom; a 3 x 4 m area with 8-15 distinct objects.
* No people or pets in frame. Avoid mirrors and large windows with glare.
* Prefer textured, matte objects; very thin objects (cables, wire racks) are hard for
  depth models.

## Camera motion (each recording 30-60 s)
* Walk slowly along a U-shaped arc around the area, phone tilted slightly down so the
  floor and the objects on it are visible.
* Keep roughly 60 % overlap between successive viewpoints; no fast pans, no stopping
  to zoom.
* Cover the whole area in both recordings, but take a **different path and height** in
  the second recording (the brief asks for different viewpoints).
* Start the second recording from a different corner than the first.

## The three scenes
| Scene | Changes between the two recordings | Purpose |
|---|---|---|
| 1 basic | move a chair ~1 m; put a backpack or box on the floor; remove a cushion or lamp | one of each change type |
| 2 occlusion / viewpoint | move a small object (cup, book); add an object partly hidden behind furniture; leave one object that is visible only from the first path **unchanged** | test "unverified" vs "removed" |
| 3 hard | swap two similar objects; rotate an object in place; move a large object (table) | matching and rotation |

## Ground truth to record
* For each scene, a `gt.yaml` listing each change: object name, type (added / removed /
  moved), and for moved objects the tape-measured displacement in metres.
* One photo of the before and after state from the same spot.
* Optional: a LiDAR scan of each state with a scanner app (e.g. 3D Scanner App or
  Polycam, export as PLY/OBJ) for metric verification.

## File naming
```
data/scenes/<scene>/before.mov
data/scenes/<scene>/after.mov
data/scenes/<scene>/gt.yaml
data/scenes/<scene>/photos/
```
Transfer by AirDrop/USB; do not re-encode or trim with apps that change the resolution.

## Capture-day tools and ground truth
Run from the repository root with the project venv (`.venv/bin/python scripts/...`); all
three are CPU only and print `--help`.

### Before leaving the room: `scripts/check_video.py`
```
python scripts/check_video.py data/scenes/<scene>/before.mov data/scenes/<scene>/after.mov [--json check.json]
```
One PASS / WARN / FAIL line per check and video, then the pair check and a `RESULT` line;
exit status 1 on any FAIL. Re-record on FAIL: unreadable file, shorter than 15 s, the two
videos in different orientations (one reconstruction pass needs one image size), or fewer
than 24 usable keyframes (the keyframes `roomdiff.ingest` would pick with the default
settings, minus blurry ones and the ones it only keeps while the camera was at rest).
WARN lines are worth a re-take when it is cheap: duration outside 30-60 s, portrait,
frame rate outside 24-60 fps, HEVC instead of H.264 (decodes fine) or HDR (10-bit / HLG /
Dolby Vision: colours shift between takes), more than 20 % of ~60 evenly spaced frames
blurry (variance of the Laplacian below 0.4 x the video's median) or soft overall, more
than 10 % of consecutive samples with a frame difference above 0.2 (fast pans: the times
are listed).

### Optional LiDAR scans
Scan each state with the "3D Scanner App" (LiDAR mode) right after its video, covering the
same area, and export the scan as a PLY point cloud:
```
data/scenes/<scene>/lidar/before.ply
data/scenes/<scene>/lidar/after.ply
```
After the pipeline has run:
```
python scripts/lidar_check.py outputs/<scene> --before data/scenes/<scene>/lidar/before.ply --after data/scenes/<scene>/lidar/after.ply [--json]
```
It registers our cloud of each recording (`viewer/<rec>.ply`, else `recon/<rec>.ply`, else
`recon/recon.npz`) to its scan with a similarity: both are gravity-aligned with +Y up, so
the search covers scale (0.80-1.25) and yaw, then refines by ICP. It prints the scale
(ours -> metres), yaw, tilt (levelling error), the residual median / p90 in the overlap
and the overlap fraction (WARN below 50 %: the scan shows another part of the room), the
before / after scale consistency (PASS within 3 %), and a table of the changes with
`translation_m` and `obb` sizes multiplied by the scale, to compare with the tape
measurements. `--json` writes `outputs/<scene>/lidar_check.json`; `--lidar-up z` reads a
Z-up export. On synthetic scans (1 cm noise, 30 % of the points dropped, the scan covering
only part of the room, a second room in the scan, a 1.5 degree tilt) it recovers the scale
within 1 % and the yaw within 2 degrees in about 10 s per recording.

### Ground truth file (`gt.yaml`)
Write `data/scenes/<scene>/gt.yaml` (template: `docs/gt_template.yaml`) while the changes
are in front of you:

| key | meaning |
|---|---|
| `scene` | scene name (optional, defaults to the folder name) |
| `notes` | free text (optional) |
| `changes` | list, one entry per changed object: |
| `- object` | short noun phrase, as a person would name it ("red office chair") |
| `- type` | `moved`, `added` or `removed` |
| `- displacement_m` | moved only, optional: tape-measured distance between the old and the new place of the object's centre |
| `- rotation_deg` | moved only, optional: approximate turn about the vertical |
| `- size_m` | optional: [length, width, height] in metres, for reference |
| `- where` | free text, e.g. "next to the window" |
| `unchanged_but_only_in_one_video` | objects that did not change but are deliberately visible in only one recording: `object`, `seen_in` (`before` or `after`) |

Unknown keys are errors (they are usually typos), as are distances or angles on objects
that did not move.

### Scoring a run: `scripts/score_gt.py`
```
python scripts/score_gt.py outputs/<scene> data/scenes/<scene>/gt.yaml [--json] [--threshold 0.6] [--scale S]
```
Every GT change is matched to an unused prediction of `changes.json` (not `rejected`) of
the same type whose label matches; among several, the most confident wins. A `moved` GT
object without such a match accepts a `removed` + `added` pair whose labels both match
(counted as found, type wrong; displacement = distance of the two centroids); a single
matching half is reported as `partial` and counts as missed. Labels match by token
overlap: lowercase, cut at a preposition ("lamp on the desk" -> "lamp"), articles,
colours, materials and sizes dropped (unless the last word), plurals made singular, common
synonyms merged (chair / seat, bag / backpack, box / carton, sofa / couch, jumper /
sweater, trash can / bin, pyjamas / pajamas, cushion / pillow, mug / cup, ...); the last
word (head noun) weighs twice the others and the score is the shared weight over the
weight of the shorter label ("office chair" ~ "chair" 1.0, ~ "desk chair" 0.67, "coffee
table" vs "coffee mug" 0.33; match from 0.6). The report (markdown) lists per GT change
the matched prediction(s), the type check, the displacement error (moved; `--scale`
multiplies the predicted distances, e.g. by the LiDAR scale) and the rotation error (only
when `rotation_reliable`); the counts found / type correct / missed; the extra predictions
(false positives; unmatched `unverified` changes are listed apart, as they claim no
change); and, per `unchanged_but_only_in_one_video` object, whether it was correctly not
reported as removed / added / moved (`unverified` or absent is correct). `--json` writes
`outputs/<scene>/gt_score.json`.
