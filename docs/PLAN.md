# RoomDiff — implementation plan (frozen 2026-10-12, deadline 2026-10-16)

*This is the plan as written on day 1, kept for the record. What was actually built, and
how it did, is in the [README](../README.md) and the [design note](design_note.md); the
schedule and risk tables below are as planned, not as it went (the LiDAR check and the
benchmark subset were done, the 3DGS channel was not; the own scenes became three
tabletop pairs recorded on 13 October).*

Goal: compare two phone videos of the same room, build one shared 3D representation, and
report what was **added / removed / moved**, with 3D location, semantic label and a
natural-language description, while explicitly separating real change from viewpoint,
occlusion and incomplete observation.

## 1. Positioning (what makes this not the "standard" solution)

The obvious pipeline (seen in several public attempts) is: joint VGGT run -> point-cloud
difference -> DINO matching -> VLM names -> report. The strongest published method,
SceneDiff (ECCV 2026), fuses geometric / appearance / region scores per SAM region and
labels moved objects by feature matching. We build our own object-centric pipeline and
differentiate on five points:

1. **Evidence-based verdicts.** Every candidate change is a hypothesis judged by explicit
   evidence: free-space rays from the *other* recording (geometry), feature similarity
   (appearance) and a VLM judge looking at paired crops (semantics). Output classes are
   `added`, `removed`, `moved`, and `unverified` (space never observed by the other video).
   Each verdict carries counts of supporting observations and a confidence.
2. **Registration-free shared frame.** Both videos' keyframes go through one
   permutation-equivariant feed-forward model (Pi3X), so the shared coordinate frame is
   produced directly; no SfM, no ICP alignment step. Fallback: separate runs + Sim(3)
   on static background (VGGT-CD style).
3. **Moved objects get a rigid transform** (translation distance and rotation angle) and a
   from->to "ghost" in the viewer, not just a label.
4. **Open-vocabulary object inventory.** SAM 3 concept prompts (or SAM 2.1 + DINO
   fallback) give names at segmentation time; the VLM only has to verify and describe.
5. **Presentation + numbers.** Static web viewer (GitHub Pages) with before/after/evidence
   toggles linked to the language report; quantitative evaluation on our own
   LiDAR-verified scenes and on a subset of the public SceneDiff benchmark.

## 2. Pipeline

```
before.mov, after.mov
   |  M1 ingest: decode (PyAV), sharpness + motion keyframes (~48 each), orientation fix
   v
M2 recon: joint Pi3X on all keyframes (order-free) -> per-frame pose, depth, conf, points
   |     sanity: static-background residual; fallback = separate runs + Sim(3)
   v
M3 visibility: per-recording 3-state voxel grid {occupied, free, unknown} by ray carving
   |     candidates = occ_A & free_B (disappeared), occ_B & free_A (appeared),
   |     occ_A & unknown_B / occ_B & unknown_A (unverified); connected components + evidence
   v
M4 instances: SAM3 concept masks (fallback SAM2.1 auto) + DINO features per mask,
   |     lift to 3D, multi-view fusion -> object list per recording
   |     cross-recording matching (Hungarian on feature + size) -> moved (SE(3) via
   |     Umeyama/ICP), added, removed, unverified, static
   v
M5 semantics: Qwen3-VL-8B names/describes each change from best crops; VLM judge on
   |     paired before/after crops rejects lighting / reconstruction false positives;
   |     report.md + changes.json
   v
M6 outputs: Rerun session (.rrd), static web viewer assets, GIF, report
```

Key parameters live in one YAML config; every stage caches its outputs so later stages can
be re-run alone.

## 3. Models and licences

| Role | Primary | Fallback | Notes |
|---|---|---|---|
| Joint reconstruction | Pi3X (`yyfz233/Pi3X`, CC-BY-NC weights) | VGGT-Omega-1B-512 (gated), MapAnything (metric) | Pi3 shown best for joint two-video alignment in SceneDiff ablations |
| Segmentation | SAM 3 / 3.1 (gated, text prompts) | SAM 2.1 hiera-large automatic masks | |
| Appearance features | DINOv3 ViT-L/16 (gated) | DINOv2-with-registers-large | tiny accuracy difference per SceneDiff |
| Vision-language | Qwen3-VL-8B-Instruct (Apache-2.0, local) | — | no API keys available |
| Visualisation | Rerun 0.38 + three.js static viewer | Open3D | |

Everything runs offline on one RTX 5090 (32 GB).

## 4. Schedule

| Day | Date | Deliverable | Done when |
|---|---|---|---|
| 1 | Oct 12 | env, plan, capture protocol, M1, M2 draft | Pi3X joint run on a SceneDiff pair gives aligned clouds |
| 2 | Oct 13 | M2 validated on own videos, M3, Rerun v0 | change blobs with evidence shown in Rerun on scene 1 |
| 3 | Oct 14 | M4 (instances, matching, SE(3)), changes.json | correct added/removed/moved on scene 1 |
| 4 | Oct 15 | M5 semantics + report, M6 web viewer, 3 scenes, eval | viewer + report for all scenes; metrics table |
| 5 | Oct 16 | README, examples, design note, GitHub Pages, submit | fresh clone runs one command end-to-end |

Scope freeze: end of day 3 the core pipeline must produce `changes.json` on at least one
real scene. Stretch items are attempted only after that, in this order:
(a) SceneDiff val-subset numbers, (b) LiDAR-based metric check, (c) 3DGS render-and-compare
as an extra evidence channel and figure.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Joint reconstruction drifts / scale differs between videos | consistent capture (same phone, orientation, lighting); static-background residual check; VGGT-CD style fallback registration |
| Gated weights not approved in time | ungated fallbacks for every role, already downloaded |
| RTX 5090 (sm_120) breaks pinned deps | torch 2.10 cu128 venv; third-party repos installed with `--no-deps` |
| 23 GB RAM | keyframe cap (~48 per video), voxel grids in torch on GPU, downsampled viewer assets |
| Non-metric scale | Pi3X approximate metric; report distances with a caveat; optional tape-measure / LiDAR calibration |
| VLM hallucinated names | names must agree with SAM3 concept / CLIP vote; VLM judge sees both crops |
| Time | scope freeze day 3; stretch list ordered |

## 6. Evaluation

* Own scenes (3): hand-written ground truth (`gt.yaml`: object, type, approx displacement),
  precision / recall per type, distance error for moved objects where tape-measured.
* SceneDiff validation subset (~10 pairs): write our per-frame instance masks in their
  `object_masks.pkl` format and run their evaluation script (per-view AP, per-scene AP).
* Ablation table: without visibility reasoning, without VLM judge, geometry-only.

## 7. Repository layout

```
roomdiff/            python package (ingest, recon, visibility, instances, changes,
                     semantics, viz, eval, cli)
configs/             default.yaml
viewer/              static three.js web viewer (deployed to GitHub Pages)
examples/            small example inputs + outputs (viewer assets, report, GIF)
docs/                PLAN.md, capture_protocol.md, env_report.md, design_note.md
scripts/             helper scripts (download models, run benchmark subset)
third_party/         Pi3, sam3 (git submodules or clone instructions; gitignored for now)
```
