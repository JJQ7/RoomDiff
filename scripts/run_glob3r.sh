#!/bin/bash
# Run Glob3R (third_party/Glob3R, its own conda prefix .venv) on the keyframes of one RoomDiff
# run and convert the result to <out_dir>/glob3r_raw.npz + glob3r_meta.json.
# Usage: scripts/run_glob3r.sh <run_dir> <out_dir> [--ordered] [extra infer.py args]
#   <run_dir>/keyframes/index.json and keyframes/{before,after}/*.jpg are read, never written.
#   Input: <out_dir>/input/NNNN_<file> (all before frames in index order, then all after).
#   Default infer.py args: --unordered --save_dense --save_frames --resize_mode fit
#   --target_size T (T: largest multiple of 14 with W*H <= 255000 at the input aspect, the Pi3
#   pixel budget the RoomDiff baseline uses: 378x672 portrait, 504x504 square). Extra args are
#   appended (argparse: the last value wins). --ordered drops --unordered (then pass e.g.
#   --keyframe_mode fixed --running_keyframe --window_size 5).
#   Env: GLOB3R_MEM (10G), GLOB3R_TIMEOUT (1500 s) for scripts/run_guarded.sh -g.
# Exit status: non-zero if any step fails (run_guarded codes: 137 memory cap, 124 timeout,
# 75 too little free RAM).
# Environment (removed after the experiment, 20 GB; rebuilt without sudo as follows): a micromamba
# prefix at third_party/Glob3R/.venv with conda-forge python 3.12, cuda-toolkit 12.8 (nvcc, cudart,
# cusparse, cusolver, cublas, nvtx dev), libcudss-dev 0.7.1.4 and suitesparse 7.10.1; then pip:
# torch 2.9.0+cu128 / torchvision 0.24.0 (cu128 wheels, sm_120), numpy 1.26.4, scipy 1.13.0,
# scikit-sparse 0.4.15 (built against the prefix's SuiteSparse), requirements.txt, PyPose, the BAE
# CUDA extensions (TORCH_CUDA_ARCH_LIST=12.0, system GCC 13.3), InstantSfM, Glob3R editable, all at
# the commits pinned in Glob3R's third_party/README.md (its install_backend.sh hard-checks
# /usr/include/libcudss/12, so its three pip commands were run by hand). .tools/env.sh exports
# CONDA_PREFIX and CUDA_HOME = the prefix, PATH, CPATH / LIBRARY_PATH / LD_LIBRARY_PATH (prefix lib
# and targets/x86_64-linux), SUITESPARSE_INCLUDE_DIR / SUITESPARSE_LIBRARY_DIR, TORCH_EXTENSIONS_DIR,
# CC=gcc CXX=g++ and LDFLAGS=-Wl,-rpath,<prefix>/lib. Weights in third_party/Glob3R/weights/:
# pytorch_model.bin (hf download juneyoung9/glob3r), dino_salad.ckpt, dinov2_vitb14_pretrain.pth,
# dinov2_vitb14_reg4_pretrain.pth (sources in Glob3R's README). Glob3R commit 553096e plus
# scripts/glob3r_roomdiff.patch (per-frame depth export, --resize_mode fit, --target_size).
set -euo pipefail
[ $# -ge 2 ] || { sed -n 2,14p "$0"; exit 2; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
G3R=$ROOT/third_party/Glob3R
PY=$G3R/.venv/bin/python
RUN_DIR=$(realpath -e "$1"); OUT=$(realpath -m "$2"); shift 2
INDEX=$RUN_DIR/keyframes/index.json
[ -f "$INDEX" ] || { echo "[run_glob3r] missing $INDEX" >&2; exit 2; }
case "$OUT/" in "$RUN_DIR"/*) echo "[run_glob3r] out_dir must not be inside run_dir" >&2; exit 2;; esac
[ -x "$PY" ] && [ -f "$G3R/weights/pytorch_model.bin" ] || { echo "[run_glob3r] Glob3R env/weights missing" >&2; exit 2; }
ORDERED=0
if [ "${1:-}" = "--ordered" ]; then ORDERED=1; shift; fi
EXTRA=("$@")

mkdir -p "$OUT"
rm -rf "$OUT/input" "$OUT/glob3r"
mkdir -p "$OUT/input" "$OUT/glob3r"
# shellcheck disable=SC1091
. "$G3R/.tools/env.sh"

# Flat ordered input (copies; frames of the after video are resized to the before size when
# the two videos differ in resolution but share the aspect ratio). Prints the target size.
TS=$("$PY" -I - "$RUN_DIR/keyframes" "$OUT/input" <<'PY'
import json, os, shutil, sys
from PIL import Image
kd, out = sys.argv[1:3]
idx = json.load(open(os.path.join(kd, "index.json")))
frames = [(rec, fr["file"]) for rec in ("before", "after") for fr in idx[rec]["frames"]]
size0 = None
for i, (rec, name) in enumerate(frames):
    src = os.path.join(kd, idx[rec].get("dir", rec), name)
    dst = os.path.join(out, f"{i:04d}_{name}")
    with Image.open(src) as im:
        size = im.size
        if size0 is None:
            size0 = size
        if size == size0:
            shutil.copyfile(src, dst)
        elif abs(size[0] / size[1] - size0[0] / size0[1]) < 0.01:
            im.convert("RGB").resize(size0, Image.Resampling.BICUBIC).save(dst, quality=95)
        else:
            sys.exit(f"{src}: size {size} has a different aspect ratio than {size0}")
w, h = size0
best = 14
for L in range(14, 2000, 14):  # mirrors preprocess_image(mode="fit")
    s = round(min(w, h) * L / max(w, h) / 14) * 14
    if L * s <= 255000:
        best = L
print(best)
PY
)
N=$(ls "$OUT/input" | wc -l)
echo "[run_glob3r] $N frames, target_size $TS"

ARGS=(--data_path "$OUT/input" --save_path "$OUT/glob3r" --ckpt "$G3R/weights/pytorch_model.bin"
      --weights_dir "$G3R/weights")
DEF=(--save_dense --save_frames --resize_mode fit --target_size "$TS")
[ "$ORDERED" = 1 ] || DEF=(--unordered "${DEF[@]}")
INFER_ARGS=("${DEF[@]}" "${EXTRA[@]}")

# Device-wide VRAM sampler (baseline = other processes' use just before the run).
base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$OUT/glob3r/vram.log" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
set +e
"$ROOT/scripts/run_guarded.sh" -g -m "${GLOB3R_MEM:-10G}" -t "${GLOB3R_TIMEOUT:-1500}" -- \
    /usr/bin/time -v -o "$OUT/glob3r/time.txt" "$PY" "$G3R/infer.py" "${ARGS[@]}" "${INFER_ARGS[@]}" \
    > "$OUT/glob3r/infer.log" 2>&1
rc=$?
set -e
kill $SAMPLER 2>/dev/null || true
if [ $rc -ne 0 ]; then
    echo "[run_glob3r] infer.py failed (exit $rc); log: $OUT/glob3r/infer.log" >&2
    tail -5 "$OUT/glob3r/infer.log" >&2
    exit $rc
fi

"$PY" -I - "$OUT" "$base" "$G3R" "${INFER_ARGS[@]}" <<'PY'
import json, os, re, subprocess, sys
out, base, g3r, args = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4:]
t = open(os.path.join(out, "glob3r", "time.txt")).read()
m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", t)
parts = [float(p) for p in m.group(1).split(":")]
secs = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1)) / 2**20
vram = [float(x) for x in open(os.path.join(out, "glob3r", "vram.log")).read().split() if x.strip()]
commit = subprocess.run(["git", "-C", g3r, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
info = {"infer_args": args, "inference_s": round(secs, 1), "peak_rss_gb": round(rss, 2),
        "peak_vram_gb": round((max(vram) - base) / 1024, 2) if vram else None,
        "glob3r_commit": commit + ("+roomdiff-patch" if subprocess.run(
            ["git", "-C", g3r, "diff", "--quiet"]).returncode else "")}
json.dump(info, open(os.path.join(out, "glob3r", "run_info.json"), "w"), indent=2)
PY

"$PY" -I "$ROOT/scripts/glob3r_to_npz.py" "$OUT/glob3r" "$INDEX" "$OUT" --run_info "$OUT/glob3r/run_info.json"
echo "[run_glob3r] done: $OUT/glob3r_raw.npz"
