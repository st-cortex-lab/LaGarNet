
# Setup
1. checkout to softgym's cloth-funnel-fit branch

2. compile the softgym

2. get `cloth_funnel` data under `agent-arena/data` folder.

3. run `test_sim.py` to make the environment works.

# Collect Data

## Longsleeve

Test collecting data
```
python collect_data.py 

```

Run diffusion collect data on longsleeves
```
python collect_data.py \
    --single_agent --agent_name diffusion_policy \
    --parallel_processes 10 --trial 5000 \
    > tmp/collect-diffusion-longsleeve.txt 2>&1 &

python visualise_data.py --agent_name diffusion_policy
```

Run mask-biased collect data
```
python collect_data.py \
    --parallel_processes 10 --trial 5000 \
    > tmp/collect_data.txt 2>&1 &

python visualise_data.py
```

## Pants

```
python collect_data.py \
    --single_agent --agent_name diffusion_policy \
    --parallel_processes 10 --trial 5000 \
    --arena_name "softgym|domain:clothfunnels-real2sim-pants,task:flattening,horizon:20" \
    --agent_trained_arena_name "softgym|domain:clothfunnels-real2sim-pants,task:flattening,horizon:30" \
    > tmp/collect_data.txt 2>&1 &

python visualise_data.py \
    --agent_name diffusion_policy \
    --arena_name "softgym|domain:clothfunnels-real2sim-pants,task:flattening,horizon:30"
```


Run mask-biased collect data
```
python collect_data.py \
    --parallel_processes 10 --trial 5000 \
    --arena_name "softgym|domain:clothfunnels-real2sim-pants,task:flattening,horizon:20" \
    > tmp/mask-pants.txt 2>&1 &

python visualise_data.py \
     --arena_name "softgym|domain:clothfunnels-real2sim-pants,task:flattening,horizon:20"
```

# Evaluate Human Policy

```

cd <path-to-agent-arena>/tool

python evaluate.py --agent human-pixel-pick-and-place-1  \
   --arena  "softgym|domain:clothfunnels-real2sim-<object>,task:flattening,horizon:30"     \
   --log_dir /data/garment-flattening
```
# Note
1. add ray installation


### Collect Intial states for UR3e Environment ###

```

python cloth_funnels/tasks/generate_tasks.py \
    --mesh_category skirt.json \
    --path assets/tasks/multi-smallSkirt-train.hdf5 \
    --task unfold \
    --num_processes 8 \
    --task_difficulty hard  \
    --num_tasks 1500 \
    --randomize_direction \
    --random_translation 0.3 \
    --local_mode \
    --scale 0.4

python cloth_funnels/tasks/generate_tasks.py \
    --mesh_category skirt.json \
    --path assets/tasks/multi-smallSkirt-train.hdf5 \
    --task unfold \
    --num_processes 8 \
    --task_difficulty easy \
    --num_tasks 2000  \
    --random_translation 0.05 \
    --local_mode \
    --scale 0.4

python cloth_funnels/tasks/generate_tasks.py \
    --mesh_category skirt.json \
    --path assets/tasks/multi-smallSkirt-eval.hdf5 \
    --task unfold --num_processes 8 \
    --eval --task_difficulty hard \
    --randomize_direction \
    --random_translation 0.3 \
    --num_tasks 200 \
    --local_mode \
    --scale 0.4

python cloth_funnels/tasks/generate_tasks.py \
    --mesh_category skirt.json \
    --path assets/tasks/multi-smallSkirt-eval.hdf5 \
    --task unfold \
    --num_processes 8 \
    --eval \
    --task_difficulty easy  \
    --random_translation 0.05 \
    --num_tasks 400 \
    --local_mode \
    --scale 0.4
```

# Real-World Trials

We use `ROS2 Humble` to setup our real-robot experiments.

## 1. Set up the UR3e robot

We use `robot-control-cloth` repository to assist this process. TODO: put the link.

After making the above repository correctly installed.

In the first terminal,
```
source /opt/ros/humble/setup.sh

ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.1.15 launch_rviz:=false

```

You may need to chagne the `robot_ip` to fit your setup.

## 2. Initialised MoveIt!

In the second terminal, go to the workspace that setup up `robot-control-cloth`. Then, run the following command to intialise MoveIt!

```
source /opt/ros/humble/setup.bash
source ./rcc_build/install/setup.sh

ros2 launch rcc_qs_pnp \
    ur_robot_moveit_executable_launch.py \
    ur_type:=ur3e \
    executable:=quasi_static_pick_and_place.py \
    planner:=ompl

```

## 3. Setup LaGarNet for Real World Experiment


1. Make sure the `agent-arena-v0` root directory, swith to branch `ros2-humble`. Then, build its conda environment under this branch by following the tutorial of agent-arena.

2. Go to the root of this `LaGarNet` repo
```
. ./setup.sh real-world
```
