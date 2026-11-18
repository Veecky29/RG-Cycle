#!/usr/bin/env bash
# Start the pilot scheduler in the background (setsid nohup). Refuses if it is already running,
# already completed, or if any other process is using the GPU.
# Usage: bash pilot/launch.sh RUN_DIR [PYTHON]
set -u
RUN=$(readlink -f "${1:?usage: launch.sh RUN_DIR [PYTHON]}")
PY=${2:-python}
REPO=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$RUN/protocol.json" ] || { echo "no $RUN/protocol.json (run --init first)"; exit 2; }
for pid in $(pgrep -f "pilot/run_pilot.py"); do
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  if [[ "$cmd" == *"--run-dir $RUN"* ]]; then echo "pilot already running (pid $pid); nothing to do"; exit 0; fi
done
if [ -f "$RUN/results/status.json" ] && grep -q '"state": "completed"' "$RUN/results/status.json"; then
  echo "pilot already completed; nothing to do"; exit 0; fi
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
if [ "$busy" != "0" ]; then echo "GPU busy ($busy compute process(es)); not launching"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; exit 3; fi
cd "$REPO" && setsid nohup "$PY" -u "$REPO/pilot/run_pilot.py" --run-dir "$RUN" >> "$RUN/scheduler.out" 2>&1 < /dev/null &
sleep 8; echo "launched pilot scheduler"; tail -3 "$RUN/scheduler.out"
