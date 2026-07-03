#!/bin/bash
# Unattended completion of the CMIP6 extension:
#   1. download rounds until all 42 variables are cached (resumable)
#   2. bias correction (cmip6_prepare.py)
#   3. future projections + change maps (cmip6_project.py)
# Every step logs; failures retry in the next round.

cd "$(dirname "$0")"
source ../.venv/bin/activate

TOTAL=42
for round in $(seq 1 12); do
  n=$(ls ../data_cmip6/cache/*.nc 2>/dev/null | wc -l | tr -d ' ')
  echo "[autopilot] round ${round}: ${n}/${TOTAL} variables cached"
  if [ "$n" -ge "$TOTAL" ]; then
    echo "[autopilot] download complete"
    break
  fi
  python -u cmip6_download.py
  echo "[autopilot] round ${round} finished"
done

n=$(ls ../data_cmip6/cache/*.nc 2>/dev/null | wc -l | tr -d ' ')
echo "[autopilot] final cache count: ${n}/${TOTAL}"

# run downstream steps for whatever is complete (prepare/project skip missing)
echo "[autopilot] === bias correction ==="
python -u cmip6_prepare.py
echo "[autopilot] === projections & change maps ==="
python -u cmip6_project.py
echo "[autopilot] ALL DONE"
