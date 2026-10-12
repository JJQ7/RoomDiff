# Environment report

Date: 2026-10-12. The WSL VM restarted twice during setup. After the final restart I re-ran the downloads, the VRAM probe and smoke tests b, c, d, e, h and i under the current WSL configuration. Each job ran alone through `scripts/run_guarded.sh`, which applies a cgroup memory cap with no swap, the shared GPU lock and a timeout. GPU tests used a 10 GB cap and every job exited with rc=0. Items marked *(earlier session)* were measured before the restarts and recovered from the session log. They do not depend on the VM memory size.

## 1. System

- Host: Windows 10 Pro 19045, 31.1 GB RAM, pagefile 16-32 GB. Guest: WSL2 Ubuntu 24.04.4, kernel 6.18.40.1-microsoft-standard-WSL2.
- WSL VM: 16 GB RAM (about 13 GB available at idle) and 4 GB swap. During the first two attempts it was 24 GB and 24 GB.
- GPU: RTX 5090, 32 GB, WDDM, driver 595.79, compute capability 12.0. torch is built against CUDA 12.8.
- Python 3.12.3 venv at `/home/jjq/code/intern_humanoid/.venv`, uv 0.13.0. Disk: about 735 GB free.

## 2. Packages

| package | version | package | version |
|---|---|---|---|
| torch / torchvision | 2.10.0+cu128 / 0.25.0+cu128 | transformers | 5.19.0 |
| numpy | 2.5.2 | accelerate | 1.15.0 |
| huggingface-hub / hf-xet | 2.2.0 / 1.7.0 | safetensors / tokenizers | 0.8.0 / 0.23.3 |
| timm | 1.0.30 | open3d | 0.20.0 (needs libusb fix, see section 7) |
| rerun-sdk | 0.38.1 | av (PyAV, bundled FFmpeg, libavcodec 63.1.102) | 19.0.1 |
| opencv-python-headless | 5.0.0.93 | pycocotools | 2.0.11 |
| qwen-vl-utils | 0.0.14 | scikit-learn / scipy | 1.9.1 / 1.18.1 |
| xformers / triton | 0.0.35 / 3.6.0 | flash-attn | 2.8.1 (prebuilt cu12/torch2.10 wheel; not used by the tests) |
| pi3 | editable, third_party/Pi3 @ 9fa3ddb | sam3 | 0.1.0 editable, third_party/sam3 @ 0570b3a |

The full pin list is in `requirements.lock.txt` (`uv pip freeze`, 143 lines).

**numpy decision: kept 2.5.2.** third_party/sam3 declares `numpy<2` and Pi3 pins 1.26.4. Everything still works under 2.5.2: SAM 3 import and inference, Pi3X, the pycocotools RLE decode, Open3D, PyAV and transformers. No pin is needed.

## 3. Models

Cache: `~/.cache/huggingface/hub` (`HF_HOME` is not set). huggingface_hub 2.2 keeps file contents in a shared Xet store at `hub/blobs/<xx>/<hash>`, and each repo's `blobs/<sha256>` is a symlink into it. Plain `du -sh models--*` therefore shows only KB; measure with `du -shL models--X/snapshots`. Total store size is 40 GB. HF is logged in as `jjq77` and the gated repos are accessible.

| repo | revision | size | main files | check |
|---|---|---|---|---|
| yyfz233/Pi3X | bb1deea4d7 | 5.1G | model.safetensors (5,440,325,620 B) | sha256 OK |
| Qwen/Qwen3-VL-8B-Instruct | 0c351dd01e | 17G | 4 safetensors shards | sha256 OK |
| facebook/dinov2-with-registers-large | e4c89a4e05 | 1.2G | model.safetensors | sha256 OK |
| facebook/sam2.1-hiera-large | 665f8e2ad6 | 1.7G | model.safetensors (transformers) + sam2.1_hiera_large.pt (original) | sha256 OK |
| facebook/dinov3-vitl16-pretrain-lvd1689m | ea8dc2863c | 1.2G | model.safetensors (1,212,559,808 B) | sha256 OK |
| facebook/sam3 | 3c879f3982 | 6.5G | sam3.pt (3,450,062,241 B, official repo) + model.safetensors (3,439,938,512 B, transformers Sam3Model) | sha256 OK |
| facebook/sam3.1 | daa6319184 | 3.3G | sam3.1_multiplex.pt (3,502,755,717 B) | sha256 OK |
| facebook/VGGT-Omega (single file) | 1041e80fc0 | 4.3G | vggt_omega_1b_512.pt (4,576,706,117 B) | sha256 OK |

- Snapshot paths have the form `~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<full revision>/`.
- SAM 3 checkpoint used below: `models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt`.
- VGGT-Omega: `models--facebook--VGGT-Omega/snapshots/1041e80fc0e911235d3426b0a3d9a81075111a53/vggt_omega_1b_512.pt`. No VGGT-Omega code is installed and the brief did not ask for a smoke test.

## 4. GPU memory budget: about 22 GiB usable, not 32

- Under WSL2, CUDA allocations count against the Windows commit charge. Holding 10 GiB of VRAM cut host free commit by about 8 GB *(earlier session)*.
- Probe (1 GiB chunks, current config): OOM at **22 GiB**, with host free commit at 10.4 GB.
- On the old 24 GB VM the probe reached only 16-16.75 GiB while the guest page cache was large, and 22.75 GiB after the cache was dropped.
- Guest page cache also uses host memory. Running `posix_fadvise(DONTNEED)` over the HF blobs raised host free commit from 10.4 to 15.2 GB. While loading weights, the test scripts run this in a background thread on the checkpoint files.
- Run one GPU process at a time *(earlier session)*: when two processes ran together, the second OOMed at 6.25 GiB while the first held 10 GiB.
- Peak numbers below are `torch.cuda.max_memory_allocated` / `max_memory_reserved`. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was set by the wrapper.

## 5. Smoke tests

Test image: `third_party/Pi3/examples/room/rgb/0.png`, 434x350, a synthetic room with a folding chair, sofa, side table, mirror and window.

**(a) Pi3X, room folder (8 images)** *(earlier session)*. Loaded per `example.py` / `example_mm.py`: `Pi3X.from_pretrained("yyfz233/Pi3X")`, bf16 autocast, `load_images_as_tensor`.
- Preprocessing: resized to 560x448 (PIXEL_LIMIT 255000, sides multiples of 14).
- Load: 9.2 s; 992.5M params, fp32 weights 3.70 GiB.
- Inference: 1.00 s cold, 0.37 s warm; peak 7.63 GiB allocated / 10.68 GiB reserved.
- Outputs:
  - `points`, `local_points` and `rays`: (1,8,448,560,3)
  - `conf`: (1,8,448,560,1)
  - `camera_poses`: (1,8,4,4), camera-to-world, OpenCV convention
  - `metric`: (1,), value 0.323
- There is no intrinsics output. `pi3.utils.geometry.recover_intrinsic_from_rays_d(rays, force_center_principal_point=True)` returns K of shape (1,8,3,3).
  - Frame 0: fx=321.7, fy=321.3, cx=279.5, cy=223.5.
  - The ground truth from `examples/room/condition.npz`, scaled to 560x448, is fx=325.5, fy=322.9. Per-frame fx ranges 315.9-328.9, within about 1-3%.
- Public functions in `pi3/utils/geometry.py`: se3_inverse, get_pixel, depthmap_to_absolute_camera_coordinates, depthmap_to_camera_coordinates, homogenize_points, get_gt_warp, warp_kpts, geotrf, inv, opencv_camera_to_plucker, depth_edge, points_to_normals, normal_edge, depth_normal_edge, recover_intrinsic_from_rays_d.
- Public functions in `pi3/utils/basic.py`: load_images_as_tensor, load_multimodal_data, tensor_to_pil, array_to_pil, rotate_target_dim_to_last_axis, write_ply.
- PLY: conf>0.1 kept all pixels; after the `depth_normal_edge` filter, 1,950,115 points (97.2%), 52.7 MB. It was written to the scratchpad, which the restart wiped.
- RoPE2D warning: `pi3/models/layers/pos_embed.py` tries `from models.curope import cuRoPE2D` and falls back to a PyTorch RoPE2D. There are no curope sources and no setup.py anywhere in third_party/Pi3, so nothing can be built from the repo. The fallback is functional.

**(b) Pi3X capacity** (current config, `disable_multimodal()`, default head chunk 64, bf16 autocast)

| frames | source | resolution | tokens | cold | warm | peak alloc / reserved |
|---|---|---|---|---|---|---|
| 80 | skating.mp4, interval 1 (first 80 of 102) | 700x364 | 104,000 | 8.21 s | 6.57 s (82 ms/frame) | 19.85 / 21.44 GiB |
| 96 | skiing.mp4, interval 4 (first 96 of 140) | 672x378 | 124,416 | 10.08 s | 9.31 s (97 ms/frame) | 20.37 / 20.78 GiB |

- 120 frames was skipped as instructed. *(Earlier session, same allocator settings)*: 120 frames took 13-14 s at 19.84 GiB peak allocated.
- Peak memory comes from `Pi3X._chunked_conv_head(chunk_size=64)`, not from attention. With chunk_size=16 *(earlier session)* speed was the same, and peak allocated was:

  | frames | peak allocated (chunk 16) | time |
  |---|---|---|
  | 80 | 11.95 GiB | ~7 s |
  | 120 | 13.37 GiB | ~13 s |
  | 160 | 14.83 GiB | 22 s |
  | 200 | 16.27 GiB | 32 s |

- Recommendation: use chunk 16 for more than about 64 frames. With chunk 64, 96 frames is close to the 22 GiB ceiling.
- Process RSS stays at about 2.2 GiB.

**(c) Qwen3-VL-8B-Instruct.** `Qwen3VLForConditionalGeneration.from_pretrained(dtype=bfloat16, device_map="cuda", low_cpu_mem_usage=True)` plus `AutoProcessor`; the class exists in transformers 5.19.
- Load: 10.5 s; 8.77B params, 16.33 GiB on GPU; attention `sdpa`; process RSS about 1-2 GiB.
- Image prompt: 177 prompt tokens (pixel_values (616,1536), image_grid_thw (1,3)). Generation took 1.74 s cold and 0.56 s warm for 18 new tokens (32 tok/s); peak 16.40 GiB.
- Answer: "- Chair - Couch - Table - Mirror - Door - Wall". The "door" is actually the window.
- Text-only "Write one sentence." took 0.51 s and answered "The sun dipped below the horizon, painting the sky in hues of orange and pink."

**(d) DINOv2** (`facebook/dinov2-with-registers-large`, AutoModel and AutoImageProcessor)
- Model: Dinov2WithRegistersModel, 304M params, load 1.4 s, processor BitImageProcessor.
- patch 14, hidden 1024, `num_register_tokens=4`: skip 1 CLS + 4 registers, so patch tokens are `last_hidden_state[:, 5:]`.
- Default processor (resize, center-crop 224): last_hidden_state (1,261,1024), patch tokens (1,256,1024), a 16x16 grid. Warm latency 8.4 ms with bf16 autocast.
- No crop at 560x448: (1,1285,1024), patch tokens (1,1280,1024), a 32x40 grid, 12.7 ms; peak 1.75 GiB.

**(e) SAM 2.1** (`facebook/sam2.1-hiera-large`)
- transformers 5.19 provides Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor and Sam2VideoInferenceSession.
- `pipeline("mask-generation", model=..., device="cuda")` works and uses Sam2Model. Load 1.9 s; with default thresholds it returned 7 masks at (350,434), scores 0.918-0.993. Time 1.42 s cold, 0.79 s warm; peak 3.65 GiB.
- Single point prompt with Sam2Model and Sam2Processor, using `input_points=[[[[200,290]]]]` and `input_labels=[[[1]]]`:
  - `pred_masks` (1,1,3,256,256) and `iou_scores` [0.04, 0.94, 0.77]; `post_process_masks` gives (1,3,350,434).
  - The best mask is the chair: 9,476 px, bbox x 132-289, y 212-349. Warm latency 72 ms.

**(f) SAM 3 code.** `from sam3.model_builder import build_sam3_image_model` and `from sam3.model.sam3_image_processor import Sam3Processor` import without error under numpy 2.5.2 (5.7 s cold import).
- Public builders: build_sam3_image_model, build_sam3_video_model, build_sam3_video_predictor, build_sam3_predictor, build_sam3_multiplex_video_model, build_sam3_multiplex_video_predictor, build_tracker.
- `download_ckpt_from_hf(version="sam3"|"sam3.1")` fetches the checkpoint.
- transformers 5.19 also has native Sam3Model, Sam3Processor and Sam3VideoModel; facebook/sam3's model.safetensors is in that format.

**(g) PyAV HEVC.** `av.codec.Codec('hevc','r')` is available, and libx265 and hevc_nvenc encoders are also present. Decoding 10 frames with `thread_type="AUTO"` and converting to rgb24 ndarray:

| file | format | time per frame (incl. rgb24 conversion) |
|---|---|---|
| living_room_1_living_room_2/original_video1.mp4 | HEVC Main, 1920x1080 | 13.8 ms (conversion 6.4 ms) |
| kitchen_1_kitchen_2/original_video1.MOV | HEVC Main 10, yuv420p10le | 9.7 ms |
| bed_1_bed_2/original_video1.mov | 1920x1440 | 5.4 ms |

All three report `frame.rotation == -90`. `to_ndarray` returns the stored landscape frame, so the caller must apply the rotation (see section 6).

**(h) SAM 3 image text prompt.** `build_sam3_image_model(checkpoint_path=<snapshot>/sam3.pt, load_from_HF=False, device="cuda")` plus `Sam3Processor(model, confidence_threshold=0.5)`. Inference used bf16 autocast and TF32, as in the official notebook.
- Load: 7.0 s, 841M params, 3.33 GiB on GPU, no missing-key warnings. Peak process RSS is **7.41 GiB** during the load, because `torch.load` reads the whole 3.45 GB checkpoint into RAM and the model is built on the CPU first.
- Prompt "chair": 1 instance, score 0.941, box [131.9, 211.1, 290.5, 349.7], mask (1,1,350,434) bool, 9,262 px.
  - Latency 627 ms cold (set_image 243 ms), 119 ms warm (set_image 45 ms); peak 5.04 GiB.
- Prompt "table": 2 instances, scores 0.855 and 0.762, 82 ms. The first is the real side table; the second is the table **reflected in the mirror**.
- Output keys: backbone_out, boxes, geometric_prompt, masks, masks_logits, original_height, original_width, scores.
- numpy 2.x caused no errors. The only warning is a timm `timm.models.layers` deprecation FutureWarning.

**(i) DINOv3 ViT-L/16** (`facebook/dinov3-vitl16-pretrain-lvd1689m`, AutoModel)
- Model: DINOv3ViTModel, 303M params, load 1.5 s, processor DINOv3ViTImageProcessor (224x224 resize, no crop, ImageNet mean/std).
- patch 16, hidden 1024, 4 registers: patch tokens are `last_hidden_state[:, 5:]`.
- 224x224: (1,201,1024), patch tokens (1,196,1024), a 14x14 grid. Warm 11.0 ms.
- No crop at 560x448: (1,985,1024), patch tokens (1,980,1024), a 28x35 grid, 12.6 ms; peak 1.73 GiB.

## 6. SceneDiff benchmark *(earlier session; inspected with `python -I`, nothing executed from the dataset)*

**Layout and splits**
- Location: `data/scenediff/scenediff_benchmark/` with `data/` (12 GB), `splits/` and `vis/` (a Flask viewer; ignored).
- The zip was deleted after a CRC32 check of all 2,463 files (12.09 GB, 0 mismatches). `data/scenediff/download.log` (68 KB) remains.
- Folders are **not** named `sequence_pair_*`. There are 355 pair folders of two kinds:
  - 200 "varied" pairs named `<scene>_<i>_<scene>_<j>`, e.g. `living_room_1_living_room_2`.
  - 155 kitchen pairs in EPIC style, e.g. `P01-20240202-110250_0005_P01-20240202-161354_0017`.
- Splits, each a dict of lists:
  - `val_split.json`: varied 50 + kitchen 55 = 105.
  - `test_split.json`: varied 150 + kitchen 100 = 250.
  - The two splits do not overlap, and their union equals the 355 folders exactly.
- First 10 entries:
  - val: airhockeytable_1_airhockeytable_2, bathroom_11_bathroom_12, bathroom_1_bathroom_2, bed_3_bed_4, bedroom_1_bedroom_2, bedroom_2_bedroom_3, bedroom_2_bedroom_4, bedroom_30_bedroom_31, bedroom_3_bedroom_5, bedroom_4_bedroom_5. Val kitchen starts with P09-20240621-093545_0001_P09-20240621-093545_0004.
  - test: bathroom_3_bathroom_4, bathroom_5_bathroom_6, bathroom_7_bathroom_8, bathroom_9_bathroom_10, bed_1_bed_2, bedroom_10_bedroom_11, bedroom_12_bedroom_13, bedroom_14_bedroom_15, bedroom_16_bedroom_17, bedroom_18_bedroom_19. Test kitchen starts with P01-20240202-110250_0005_P01-20240202-161354_0017.

**Files per pair**
- 344 folders have `inference_data.pkl, original_video{1,2}.{mp4|MOV|mov}, review_meta.json, segments.pkl, video1.mp4, video2.mp4`, and 1 more has the same set without `inference_data.pkl`.
- 10 folders have only `original_video{1,2}.MOV` and `segments.pkl`.
- A few have extras: `*_compressed*.mp4`, `copy.pkl`, `frame_visualizations/`.
- Original video extensions: 188 .mp4, 149 .MOV, 18 .mov.
- **There is no `metadata.json`.** The semantic metadata is in `segments.pkl`.
- `review_meta.json` only holds annotation review info. Example from living_room_1_living_room_2: `{"sessionId": "20250316_091121_20250316_091244_2025-05-08_22-40-17", "reviewerName": "YQ", "reviewerComments": "", "timestamp": "2025-05-08T23:25:16.105Z", "objectReviews": [{"index": 0, "comment": "", "needsReannotation": false, "reviewed": true}, ... 6 entries]}`.

**segments.pkl** (pickle protocol 4)
- A static opcode scan found no globals, so it loads safely with an unpickler that forbids all globals.
- Root dict: `scene_type` (str), `light_change` (bool), `comments` (dict), `objects` (list of dict), `video1_objects` and `video2_objects`.
  - `video{1,2}_objects`: dict of obj_idx (int) -> dict of frame index (str) -> COCO RLE `{"size": [H, W], "counts": str}`.
  - `objects[i]` keys: label, in_video1, in_video2, video1_frame_idx, video1_status, video2_frame_idx, video2_status, original_obj_idx, deformability. 62 objects lack deformability.
- Example, living_room_1_living_room_2 (6 objects; light_change=True):
  - wallet_1 and bag_1: only in video 1.
  - ladder_1, right_curtain and left_curtain: in both.
  - hairbrush_1: only in video 2.
  - video1_objects has tracks for objects 0-4 (63-92 frames each); video2_objects for objects 2-5.
  - Decoding `video1_objects[0]['0']` with pycocotools: size [1920,1080] gives mask (1920,1080) uint8, area 2842, bbox [801,885,63,73].
- Whole-dataset statistics:
  - Objects: 1,017 total, mean 2.86 per pair, max 17. 10 pairs have no objects.
  - Presence: removed (T,F) 314, added (F,T) 284, present in both (T,T) 419.
  - Deformability: rigid 891, deformable 64, missing 62.
  - light_change is True in 19 pairs.
  - scene_type labels are inconsistent ('office'/'Office', 'living_room'/'living room') and need normalising.
  - RLE [H,W] per track: 1024x1024 674, 1920x1080 564, 1280x720 145, 720x1280 25, 1080x1920 11, 1920x1440 10, 3840x2160 5, 1918x1028 3.

**inference_data.pkl**
- References torch and collections globals (`torch._utils._rebuild_tensor_v2`, `torch.storage._load_from_bytes`, `OrderedDict`). It was inspected only with an allow-list unpickler.
- Contents: the SAM2 annotation prompts per video: `obj_idx_to_id`, `point_inputs_per_obj{obj:{frame:{point_coords, point_labels}}}`, `frames_dir`, `frame_names`.

**Videos (PyAV)**

| pair | file | codec | stored size | fps | duration | frames |
|---|---|---|---|---|---|---|
| living_room_1_living_room_2 | original_video1.mp4 | HEVC Main | 1920x1080, rot -90 | 29.92 | 9.39 s | 281 |
| | original_video2.mp4 | HEVC Main | 1920x1080, rot -90 | 30.02 | 11.36 s | 341 |
| | video1.mp4 / video2.mp4 | mpeg4 | 1080x1920 upright | 30 | 9.4 / 11.37 s | 282 / 341 |
| kitchen_1_kitchen_2 | original_video1.MOV | HEVC Main 10 (10-bit) | 1920x1080, rot -90 | 29.98 | 10.14 s | 304 |
| | original_video2.MOV | HEVC Main 10 (10-bit) | 1920x1080, rot -90 | 29.98 | 12.84 s | 385 |
| bed_1_bed_2 | original_video1.mov | HEVC Main yuvj420p | 1920x1440, rot -90 | 30.0 | 4.00 s | 120 |
| | original_video2.mov | HEVC Main yuvj420p | 1920x1440, rot -90 | 30.0 | 3.97 s | 119 |
| P01-20240204-120411_0017_P01-20240204-120411_0018 | original_video1.mp4 | H.264 High | 1024x1024 | 10 | 6.0 s | 60 |
| | original_video2.mp4 | H.264 High | 1024x1024 | 10 | 5.3 s | 53 |
| | video1.mp4 / video2.mp4 | mpeg4 | 1024x1024 | 30 | 6.0 / 5.3 s | 180 / 159 |

**The masks index the re-encoded `video1.mp4` / `video2.mp4` frames (upright, 30 fps), not the originals.**
- living_room_1_living_room_2: mask frame keys reach 281. video1.mp4 has 282 frames; the original has 281.
- P01-20240204-120411_0017_P01-20240204-120411_0018: video2 mask keys reach 158. video2.mp4 has 159 frames (30 fps); the original has 53 (10 fps).
- The RLE sizes are portrait (H=1920, W=1080), which matches the upright re-encodes.
- If the HEVC originals are used, rotate frames by `frame.rotation` and map frame indices by timestamp.

## 7. Problems and workarounds

1. **Open3D failed to import** (`libusb-1.0.so.0` missing; no sudo). I fetched Ubuntu's `libusb-1.0-0` 2:1.0.27-1 with `apt-get download`, which needs no root. Then I extracted `libusb-1.0.so.0.4.0` and its symlink into `.venv/lib/`. Open3D's RUNPATH `$ORIGIN/../../../` resolves there. Import, voxel downsample and PLY write/read all work, and the import was re-checked after the restart. If the venv is recreated, this has to be done again.
2. **WSL2 froze or restarted twice** under concurrent heavy jobs, on the 24 GB VM with 24 GB swap. The VM is now 16 GB + 4 GB swap, and every download and model job goes through `scripts/run_guarded.sh`, strictly one at a time.
3. **Usable VRAM is about 22 GiB** because of host commit (section 4). On the old config, the Pi3X 80- and 120-frame runs first OOMed at about 16 GiB. They were fixed by dropping the guest page cache, `expandable_segments`, and the smaller VM. For longer sequences, reduce Pi3X's conv-head chunk size.
4. **The SAM 3 load peaks at 7.4 GiB process RSS**: the full `torch.load` of sam3.pt plus building the model on the CPU. This fits the 10 GB cap. For tighter caps, load the state dict with `torch.load(..., mmap=True)` after moving the model to CUDA.
5. **`time.time()` jumped under WSL2** and gave negative durations in the first Pi3X run. All timings use `time.perf_counter()`.
6. **The brief's SceneDiff assumptions did not match the data**: there are no `sequence_pair_*` folders and no `metadata.json`. The real layout is documented in section 6. The zip was verified and deleted.
7. **Plain `du` under-reports the HF cache**, because files live in the shared Xet store behind symlinks. Measure with `du -shL`. The SHA-256 of every large file matches its LFS hash.
8. **The Pi3X CUDA RoPE2D is not available** (no curope sources in the repo). The PyTorch fallback works; speed is as measured above.
9. **Harmless warnings**: SAM2 config `memory_attention_rope_theta` deprecation, timm `timm.models.layers` deprecation, and Pi3's RoPE2D fallback message.
10. **flash-attn 2.8.1 is installed in the venv**, but Qwen3-VL defaulted to `sdpa`. `attn_implementation="flash_attention_2"` was not tested.

## 8. Files

- `requirements.lock.txt`: `uv pip freeze` output.
- `.gitignore`: written earlier and verified with `git check-ignore`.
- This report.
- Throwaway test scripts are in the session scratchpad under `/tmp/claude-1000/.../scratchpad/`: common.py, vram_probe.py, pi3x_capacity.py, qwen3vl_test.py, dino_test.py, sam2_test.py, sam3_test.py, hf_download.py, sha_check.py, drop_cache.py. /tmp is cleared on restart.
