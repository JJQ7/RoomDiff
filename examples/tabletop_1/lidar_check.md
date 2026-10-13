before: ours outputs/tabletop_1/viewer/before.ply (382,727 -> 206,029 points at 2 cm), scan data/scenes/tabletop_1/lidar/before.obj (88,095 -> 48,154)
  scale 0.5620 (ours -> metres), yaw 152.2 deg, tilt 2.1 deg; residual median 0.0288 m, p90 0.0776 m over the overlap (88% of our points within 10 cm, 66% within 5 cm); 17.3 s
after: ours outputs/tabletop_1/viewer/after.ply (362,482 -> 188,973 points at 2 cm), scan data/scenes/tabletop_1/lidar/after.obj (182,181 -> 27,931)
  scale 0.5554 (ours -> metres), yaw -129.9 deg, tilt 1.9 deg; residual median 0.0245 m, p90 0.0658 m over the overlap (89% of our points within 10 cm, 72% within 5 cm); 15.3 s
PASS  scale consistency: before 0.5620, after 0.5554, differ by 1.2% (want < 3%)

changes.json in metres (x 0.5587):

| id | type | label | distance (m) | metric distance (m) | size w x h x d (m) | metric size (m) |
|---|---|---|---|---|---|---|
| 1 | moved | white product box | 0.79 | 0.44 | 0.28 x 0.33 x 0.13 | 0.16 x 0.19 x 0.07 |
| 2 | moved | silver trash can | 0.19 | 0.10 | 0.22 x 0.29 x 0.55 | 0.12 x 0.16 x 0.31 |
| 3 | removed | tea canister | - | - | 0.20 x 0.25 x 0.19 | 0.11 x 0.14 x 0.11 |
| 9 | unverified | wooden stool | - | - | 0.57 x 0.17 x 3.40 | 0.32 x 0.10 x 1.90 |
| 10 | unverified | black office chair | - | - | 0.82 x 0.85 x 0.88 | 0.46 x 0.47 x 0.49 |
| 11 | unverified | blue suitcase | - | - | 1.00 x 0.19 x 3.72 | 0.56 x 0.11 x 2.08 |
| 12 | unverified | wooden stool | - | - | 1.94 x 0.17 x 1.55 | 1.08 x 0.10 x 0.86 |
