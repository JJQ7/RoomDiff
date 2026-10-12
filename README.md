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

# full pipeline: ingest -> recon -> visibility -> instances -> changes -> describe -> viz -> viewer
roomdiff run --before before.mov --after after.mov --out outputs/scene1
# single stages on an existing output folder, e.g. the semantics stage and the viewer
python -m roomdiff.cli describe --out outputs/scene1
python -m roomdiff.cli viewer --out outputs/scene1
```

- **Semantics and report** (`describe`): Qwen3-VL-8B (local, bf16, about 16.6 GiB VRAM,
  30-60 s per scene) names every change from its crops, judges it on paired before/after
  crops (wall, floor, mirror-reflection and shadow regions are moved to a `rejected` list
  with the reason), merges changes that were split across two candidates (a removed and an
  added part of the same object become one `moved` change), and writes one-sentence
  descriptions, a summary, `<out>/report.md` and the VLM log `<out>/semantics/log.jsonl`.
  Settings: `configs/default.yaml: semantics`.

- **Diagnostics**: `python scripts/plot_align.py outputs/scene1` renders top-down and side views of the before/after clouds (`<out>/recon/<run>_{top,side}.png`) to check the alignment by eye.

## Web viewer

A static three.js viewer (source in `viewer/`, no build step; three.js is loaded from
unpkg, so it needs an internet connection) shows the before/after clouds with the change
boxes, a per-change list with evidence and crops, before/after/both and tint toggles.

```bash
# writes outputs/scene1/viewer/: before.ply, after.ply (<= 400k points each), changes.json,
# crops/, scene.json and a copy of the viewer app
python -m roomdiff.cli viewer --out outputs/scene1 [--target-points 400000]
# browsers block data loading from file:// pages, so serve the folder:
cd outputs/scene1/viewer && python -m http.server 8000   # open http://localhost:8000/
```

Keys: N / P next / previous change, R reset view. `changes.json` is taken from
`<out>/changes.json`, or generated from `<out>/visibility/candidates.json` (geometry-only
regions) when the change stage has not run yet.

GitHub Pages layout (several scenes, one copy of the app):

```
docs/viewer/index.html, app.js, style.css       # copied from viewer/
docs/viewer/scenes/<name>/                      # contents of outputs/<name>/viewer/
```

Open `https://<user>.github.io/<repo>/viewer/?scene=scenes/<name>`. Without `?scene=` the
viewer loads `./scene.json` from its own folder. `*.ply` is gitignored except under
`examples/`, so the scene PLYs under `docs/viewer/` need a `.gitignore` exception.
