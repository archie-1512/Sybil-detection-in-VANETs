#!/bin/bash
# Regenerates the SUMO network + traffic demand from scratch.
# Run this once (or whenever you want a different grid size / traffic volume).
set -e
cd "$(dirname "$0")"

if [ -z "$SUMO_HOME" ]; then
    export SUMO_HOME=$(python3 -c "import sumo,os; print(os.path.dirname(sumo.__file__))")
fi
export PATH="$SUMO_HOME/bin:$PATH"

echo "Using SUMO_HOME=$SUMO_HOME"

# 2x2 grid of intersections, 200m edges, 1 lane per direction
netgenerate --grid --grid.number=2 --grid.length=200 --default.lanenumber=1 \
    --output-file=grid.net.xml --no-turnarounds

# Random traffic demand: vehicles depart between t=0 and t=600s, avg 1 every 0.8s
python3 "$SUMO_HOME/tools/randomTrips.py" -n grid.net.xml -o trips.trips.xml \
    --begin 0 --end 600 --period 0.8 --seed 42 --fringe-factor 10

echo "Network + trips generated."
