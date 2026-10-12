#!/bin/bash
# Memory watchdog for WSL2: if MemAvailable drops below LIMIT_GB, kill the largest
# python process (so the VM never thrashes into swap). Run once in the background:
#   nohup scripts/memwatch.sh > /tmp/roomdiff_memwatch.log 2>&1 &
LIMIT_GB=${1:-2}
while true; do
  avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$avail" -lt $((LIMIT_GB*1048576)) ]; then
    pid=$(ps -eo pid,rss,comm --sort=-rss | awk '$3 ~ /python/ {print $1; exit}')
    if [ -n "$pid" ]; then
      echo "$(date +%T) available=$((avail/1048576))G < ${LIMIT_GB}G: killing python pid $pid ($(ps -o rss= -p $pid | awk '{printf "%.1fG",$1/1048576}'))"
      kill -9 "$pid"
    fi
  fi
  sleep 2
done
