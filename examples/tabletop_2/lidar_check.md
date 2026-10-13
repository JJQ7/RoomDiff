before: ours outputs/tabletop_2/viewer/before.ply (218,436 -> 62,956 points at 2 cm), scan data/scenes/tabletop_2/lidar/before.obj (182,181 -> 27,931)
  scale 0.5505 (ours -> metres), yaw -25.2 deg, tilt 1.6 deg; residual median 0.0166 m, p90 0.0362 m over the overlap (95% of our points within 10 cm, 90% within 5 cm); 13.3 s
after: ours outputs/tabletop_2/viewer/after.ply (356,236 -> 107,726 points at 2 cm), scan data/scenes/tabletop_2/lidar/after.obj (162,450 -> 24,716)
  scale 0.5464 (ours -> metres), yaw -93.2 deg, tilt 1.8 deg; residual median 0.0197 m, p90 0.0461 m over the overlap (97% of our points within 10 cm, 89% within 5 cm); 16.3 s
PASS  scale consistency: before 0.5505, after 0.5464, differ by 0.7% (want < 3%)

changes.json in metres (x 0.5485):

| id | type | label | distance (m) | metric distance (m) | size w x h x d (m) | metric size (m) |
|---|---|---|---|---|---|---|
| 1 | removed | scissors | - | - | 0.76 x 0.10 x 0.23 | 0.41 x 0.05 x 0.13 |
| 2 | added | tea canister | - | - | 0.20 x 0.24 x 0.20 | 0.11 x 0.13 x 0.11 |
| 3 | added | green bucket | - | - | 0.43 x 0.28 x 0.24 | 0.24 x 0.15 x 0.13 |
| 8 | unverified | blue suitcase | - | - | 0.52 x 0.91 x 0.95 | 0.29 x 0.50 x 0.52 |
