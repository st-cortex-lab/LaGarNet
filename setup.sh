cd ../agent-arena-v0
. ./setup.sh

if [ -d "../softgym" ]; then
  cd ../softgym
  . ./setup.sh
else
  echo "Directory ../softgym does not exist. Skipping."
fi

cd ../LaGarNet

export PYTHONPATH=${PWD}:$PYTHONPATH