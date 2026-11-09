#!/bin/bash
# Usage:  source ./setup.sh              # simulation
#         source ./setup.sh real-world   # plus the UR3e ROS 2 workspace
#
# actoris_harena is installed into the active environment (README, section II),
# so it needs no sourcing; `import actoris_harena` sets its own paths.

conda activate "${LAGARNET_CONDA_ENV:-harena-sim}"

if [ -d "../softgym" ]; then
  cd ../softgym
  . ./setup.sh
  cd ../LaGarNet
else
  echo "Directory ../softgym does not exist. Skipping."
fi

export PYTHONPATH=${PWD}:$PYTHONPATH

if [ "$1" == "real-world" ]; then
  cd ../ws_ur3e
  source ./agar_build/install/setup.sh
  cd src/robot_control_cloth
  . ./setup.sh
  cd ../../../LaGarNet
fi
