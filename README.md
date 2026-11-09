<h1> LaGarNet: Goal-Conditioned Recurrent State-Space Models for Pick-and-Place Garment Flattening </h1>

Halid Abdulrahim Kadi and Kasim Terzić \
Unviersity of St Andrews

<div align="center">
    <a href="" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
    <a href="" target="_blank">
    <img src="https://img.shields.io/badge/Page-LaGarNet-blue" alt="Project Page"/></a>
</div>
</h2>

## I. Overview

This repository provide simulation and real-world experiments for LaGarNet robot learning controller for pick-and-place flattening. In simulation, we work on 

## II. Installation & Setup

Put `LaGarNet`, [`actoris_harena`](https://github.com/st-cortex-lab/actoris_harena) and
[`softgym`](https://github.com/st-cortex-lab/softgym) side by side in one directory.

1. Create the environment from actoris_harena's pinned requirements (`main` branch). This replaces
   the old `agent-arena-v0` install; LaGarNet now imports `actoris_harena`.

    ```
    conda create -n harena-sim python=3.10 -y && conda activate harena-sim
    pip install -r ../actoris_harena/requirements/lagarnet.txt
    pip install -e ../actoris_harena --no-deps
    ```

    `setup.sh` activates `harena-sim`; set `LAGARNET_CONDA_ENV` to use another environment.

2. Build `pyflex` from the `py3.10` branch of `softgym`, following its README. This is only for
   simulation experiments and policy training; real-world experiments do not need it.
    * For simulation, the garment initial states go under `actoris_harena/../data/cloth_funnel/init_states`
      (that is, a `data/cloth_funnel/init_states` directory beside the `actoris_harena` checkout): the
      `multi-<garment>-{train,eval}.hdf5` files and their `clothfunnels-<garment>-{train,eval}.json` keys.
    * On a hybrid-GPU laptop or a cluster node, `export __GLX_VENDOR_LIBRARY_NAME=nvidia __NV_PRIME_RENDER_OFFLOAD=1`
      first, or pyflex falls back to Mesa and crashes at start-up.

3. We use the `ros2-update` branch of [`robot-control-cloth`](https://github.com/halid1020/robot_control_cloth) repository to assist this process; please clone and build it under `ws_ur3e/src` directory, where `ws_ur3e` folder should be in parallel with `LaGarNet` and `actoris_harena` folders.


## III. Training Diffusion Policies & LaGarNet

1. For training the policies without data collection, we provide the pre-collected dataset [here](). Please unzip them under `LaGarNet/data/datasets` directory. **TODO: make sure this step works.**

2. If you want to collect your own dataset to train diffusion policies, go to `LaGarNet` directory

    ```
    . ./setup.sh

    cd data

    # e.g. Collecting 50 episodes of human policy for `longsleeve`
    python collect_data.py \
        --trials 50 \
        --parallel_processes 4 \
        --agent_name "human-pixel-pick-and-place-1" \
        --arena_name "softgym|domain:clothfunnels-realadapt-longsleeve,task:flattening,horizon:20" \
        --data_path <name-of-the-dataset> 
    
    # This dataset will be saved under `LaGarNet/data/datasets`
    ```

3. If you want to collect your own dataset to train `LaGarNet` policy, go to `LaGarNet` directory
    ```
    . ./setup.sh

    cd data

    ## Collecting 5000 episodes of mask-biased-random policy in the given arena name
    python collect_data.py \
        --trials 5000 \
        --parallel_processes 4 \
        --agent_name "oracle-garment|mask-biased-pick-and-place" \
        --arena_name "softgym|domain:clothfunnels-realadapt-longsleeve,task:flattening,horizon:20" \
        --data_path <name-of-the-dataset> 
    
    # This dataset will be saved under `LaGarNet/data/datasets`
    
    # For collecting trajectory using diffusion policies
    python collect_data.py \
        --trials 5000 \
        --parallel_processes \
        --agent_name diffusion_policy \
        --agent_config_name "human-50"
        --arena_name "softgym|domain:clothfunnels-realadapt-longsleeve,task:flattening,horizon:20" \
        --data_path <name-of-the-dataset> \
        --single_agent

    ```

    * We also provide `compress_data.py` to help you to reduce the image resolutions from 128x128 to 64x64 for fastening the training. 
    
    * `combine_data.py` can help you to get different mixture of datasets. 

4. Train policies, go to the root directory of `LaGarNet`.

    ```
    . ./setup.sh

    # To train the `human-50` of diffusion policy for `longsleeve` garments
    python train_policy.py \
        --agent diffusion_policy \
        --config human-50 \
        --object longsleeve

    # To train the `final` version of all-garment LaGarNet policy
    python train_policy.py \
        --agent gc_rssm 

    
    # To train the `task-specific` version of LaGarNet policy for `longsleeve` garments
    python train_policy.py \
        --agent gc_rssm \
        --object longsleeve \
        --config task-specific

    # To train the `all-garment` version of PlaNet-ClothPick policy
    python train_policy.py \
        --agent rssm \
        --config all-garment

    # To train the `task-specific` version of PlaNet-ClothPick policy
    python train_policy.py \
        --agent rssm
        --object longsleeve
        --config task-specific

    # Train results will be saved under `LaGatNet\test_results`

    ```


## IV. Evaluate Policies in Simulation

1. After training or downloading the provided models, go to the root directory of `LaGarNet`
    ```
    . ./setup.sh

    python eval_policy_in_all_clothfunnel_envs.py \
        --agent <agent-type> \
        --config <config-type>
    ```


## V. Real-World Trials


1. Open the first terminal, and run 
    ```
    source /opt/ros/humble/setup.sh

    ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.1.15 launch_rviz:=false

    ```

    You may need to chagne the `robot_ip` to fit your setup.

2. On the robot padel, start the controlling program to build communication; you should able to initalise under `Program/URCaps/External Control` on the padel and you can start the program by clicking the play botton on the pedal.

3. In the second terminal, go to the workspace `ws_ur3e` that setup up `robot-control-cloth`. Then, run the following command to intialise MoveIt!

    ```
    source /opt/ros/humble/setup.bash
    source ./rcc_build/install/setup.sh

    ros2 launch rcc_qs_pnp \
        ur_robot_moveit_executable_launch.py \
        ur_type:=ur3e \
        executable:=quasi_static_pick_and_place.py \
        planner:=ompl

    ```

4. Setup `LaGarNet` for real-world experiment. 


    * Activate the environment from section II, and install the ROS 2 Humble packages robot-control-cloth needs.

    * In the third terminal, go to the `LaGarnet` repo, and run
        ```
        . ./setup.sh real-world
        ```
    
    * If you do not want to train the models from scratch, please download [`models`](), and unzip it under `LaGarNet/test_results`.

5. Run a real-world experiment

    ```
    python ur3e_control.py --agent gc_rssm --config final

    ```
    **TODO: make the the model downloadable.


## VI. Collect Intial states for UR3e Environment

We recommend use the provided initial states for synthetic garments. However, if you insist you would like to generate.

1. You need to clone and install `cloth_funnels` repository; this recommended to be in pararallel with `LaGarNet`.

2. Go to `LaGarNet`


```
. ./setup.sh


python clothfunnels/generate_tasks.py \
    --mesh_category longsleeve.json \
    --path assets/tasks/multi-longsleeve-eval.hdf5 \
    --task unfold \
    --num_processes 8 \
    --task_difficulty hard  \
    --num_tasks 30 \
    --randomize_direction \
    --random_translation 0.3 \
    --local_mode \
    --scale 0.8

python cloth_funnels/generate_tasks.py \
    --mesh_category longsleeve.json \
    --path assets/tasks/multi-longsleeve-train.hdf5 \
    --task unfold \
    --num_processes 8 \
    --task_difficulty hard \
    --num_tasks 1000  \
    --random_translation 0.05 \
    --local_mode \
    --scale 0.8
```


