# RoomDiff — what changed in the room?

RoomDiff compares two phone videos of the same room, recorded at different times and
from different viewpoints, and reports which objects were **added**, **removed** or
**moved**, with their 3D location, a semantic label and a short description. Both
videos are reconstructed into one shared 3D frame; every reported change is backed by
explicit geometric, appearance and visual-language evidence, and changes in space the
other video never observed are reported as **unverified** instead of being guessed.
See [docs/PLAN.md](docs/PLAN.md) for the design.

Status: work in progress (day 1 of 5).

## Install

```bash
# TODO: environment setup (torch cu128 + requirements), see docs/env_report.md
pip install -e .
```

## Run

```bash
# Implemented so far: video ingest + keyframe selection
python -m roomdiff.cli ingest --before before.mov --after after.mov --out outputs/scene1
# any value in configs/default.yaml can be overridden (or use --config scene.yaml)
python -m roomdiff.cli ingest ... --set ingest.num_keyframes=32 --set ingest.max_side=768

# TODO: full pipeline
roomdiff run --before before.mov --after after.mov --out outputs/scene1
```
