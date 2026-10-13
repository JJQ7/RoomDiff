before: ours outputs/tabletop_3/viewer/before.ply (369,203 -> 101,604 points at 2 cm), scan data/scenes/tabletop_3/lidar/before.obj (162,450 -> 24,716)
  scale 0.5512 (ours -> metres), yaw 141.0 deg, tilt 1.0 deg; residual median 0.0185 m, p90 0.0439 m over the overlap (98% of our points within 10 cm, 90% within 5 cm); 12.8 s
after: ours outputs/tabletop_3/viewer/after.ply (234,974 -> 59,701 points at 2 cm), scan data/scenes/tabletop_3/lidar/after.obj (224,490 -> 33,404)
  scale 0.5712 (ours -> metres), yaw 85.2 deg, tilt 0.8 deg; residual median 0.0178 m, p90 0.0388 m over the overlap (100% of our points within 10 cm, 96% within 5 cm); 11.6 s
WARN  scale consistency: before 0.5512, after 0.5712, differ by 3.6% (want < 3%)

changes.json in metres (x 0.5611):

| id | type | label | distance (m) | metric distance (m) | size w x h x d (m) | metric size (m) |
|---|---|---|---|---|---|---|
| 1 | moved | blue laundry detergent | 0.53 | 0.30 | 0.13 x 0.42 x 0.28 | 0.07 x 0.23 x 0.16 |
| 2 | removed | canned food | - | - | 0.19 x 0.24 x 0.20 | 0.11 x 0.14 x 0.11 |
| 7 | unverified | blue suitcase | - | - | 0.26 x 0.72 x 0.67 | 0.15 x 0.40 x 0.38 |
| 8 | unverified | donna butter cookies | - | - | 0.21 x 0.23 x 0.20 | 0.12 x 0.13 x 0.11 |
