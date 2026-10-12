#!/bin/bash
# Run one heavy job under a hard memory cap, with no swap, a GPU lock and a timeout.
# Usage: scripts/run_guarded.sh [-m MemoryMax] [-t seconds] [-g] -- <command...>
#   -m  memory cap for the whole job incl. its page cache (default 12G)
#   -t  timeout in seconds (default 1200)
#   -g  take the shared GPU lock (only one GPU job at a time)
# A job that exceeds the cap is OOM-killed inside its own cgroup (exit 137); the rest of
# the machine is unaffected. This is what keeps WSL2 from thrashing into swap.
set -u
MEM=12G; TMO=1200; GPU=0
while getopts "m:t:g" opt; do case $opt in m) MEM=$OPTARG;; t) TMO=$OPTARG;; g) GPU=1;; esac; done
shift $((OPTIND-1)); [ "${1:-}" = "--" ] && shift
LOCK=${ROOMDIFF_GPU_LOCK:-/tmp/roomdiff_gpu.lock}
LOG=${ROOMDIFF_GUARD_LOG:-/tmp/roomdiff_guard.log}
avail=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
echo "$(date +%T) start mem=$MEM timeout=$TMO gpu=$GPU avail=${avail}G boot=$(uptime -s) :: $*" >> "$LOG"
if [ "$avail" -lt 6 ]; then echo "[guard] only ${avail}G RAM available, refusing to start" >&2; exit 75; fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
run() { systemd-run --user --scope --quiet -p MemoryMax="$MEM" -p MemorySwapMax=0 -p MemoryHigh="$MEM" -- timeout "$TMO" "$@"; }
if [ "$GPU" = 1 ]; then flock "$LOCK" -c "$(printf '%q ' systemd-run --user --scope --quiet -p MemoryMax="$MEM" -p MemorySwapMax=0 -- timeout "$TMO" "$@")"; else run "$@"; fi
rc=$?
echo "$(date +%T) end rc=$rc avail=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)G :: $*" >> "$LOG"
[ $rc -eq 137 ] && echo "[guard] job killed: exceeded memory cap $MEM" >&2
[ $rc -eq 124 ] && echo "[guard] job killed: exceeded timeout ${TMO}s" >&2
exit $rc
