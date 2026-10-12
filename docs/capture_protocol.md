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
