"""
This script is designed to collect one-picker pick-and-place
longsleeve flattening data from simulation. 

The special engineered policies include the following:
    - border-biased-pick-and-place (30%)
    - cloth-mask-random-pick-and-place (40%)
    - diffusion-expert (20%)
    - mix-policy (10%)
In order to include policies that can be used to achieve reasonablly well flattening trajectories. 
We also used trained diffusion policy trained by human data.

We eliminiate the purly random policy, as it is not useful for achieving the task. We will use
cloth-mask filter on the learned policy anyway.

In this collection, we aim to collect 35,000 trajectories with 30 steps horizon for each.
We will run 4 parallel processes to speed up the collection.

The transitional data includes:
    - RGB images
    - Depth images
    - Masks
    - particle positions
    - Actions
    - all possible rewards in the literature
    - step-wise evaluations

In order to make it run correctly, we need to have the following:
    - Real2Sim environment based on cloth-funnel's env.
    - a realitivly well-performned diffusoin policy.
    - all possible rewards produces correct values.
    - make sure step-wise collected data is correct.
    - make sure planet-clothpick can run a small amount of data.

"""
import ray
import numpy as np
import argparse
import cv2
import os
from dotmap import DotMap
import actoris_harena as ag_ar
from actoris_harena.utilities.perform_parallel \
    import setup_arenas
from actoris_harena.utilities.perform_parallel \
    import step_arenas

from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset
from planet_clothpick_plus_plus.utils import obs_config, action_config, reward_names, evaluation_names
from tqdm import tqdm
# arena_name = 'softgym|domain:cloth-funnel-longsleeve,task:flattening,horizon:30'
# # agent_name = 'oracle-garment|border-biased-pick-and-place'
# agent_name = 'oracle-garment|mask-biased-pick-and-place'
# agent_config = DotMap({
#     'oracle': True
# })
# agent_probs = [0.3, 0.4, 0.2, 0.1]
# trials = 10 #35000
# paralle_processes = 4

def get_actions(agents, arena_ids, ready_infos, actions):
    ready_actions = []
    for i, info in enumerate(ready_infos):
        idx = arena_ids.index(info['arena_id'])
        print('arena_id', info['arena_id'])
        print('idx', idx)
        print('len agents', len(agents))
        ready_actions.append(agents[idx].act([info], update=True)[0]) 
        # update the internal state with last info and last action first #
        # regarding the corresponding arena_id, then act to give action.
    
    for info, a in zip(ready_infos, ready_actions):
        arena_id = info['arena_id']
        idx = arena_ids.index(arena_id)
        action = a['norm-pixel-pick-and-place']
        action = np.stack([action['pick_0'], action['place_0']])
        actions[idx]['norm-pixel-pick-and-place'].append(action)

    return ready_actions

def update_observations(observations, idx, info):
    #   print('rgb shape', info['observation']['rgb'].shape)
    rgb = cv2.resize(info['observation']['rgb'], (128, 128))
    depth = cv2.resize(info['observation']['depth'], (128, 128))
    mask = cv2.resize(info['observation']['mask'].astype(np.float32), (128, 128))
    mask = mask > 0.9

    goal_rgb = cv2.resize(info['goal']['rgb'], (128, 128))
    goal_depth = cv2.resize(info['goal']['depth'], (128, 128))
    goal_mask = cv2.resize(info['goal']['mask'].astype(np.float32), (128, 128))
    goal_mask = goal_mask > 0.9

    observations[idx]['rgb'].append(rgb)
    observations[idx]['depth'].append(depth)
    observations[idx]['mask'].append(mask)
    observations[idx]['goal_rgb'].append(goal_rgb)
    observations[idx]['goal_depth'].append(goal_depth)
    observations[idx]['goal_mask'].append(goal_mask)
    for key in reward_names:
        observations[idx][key].append(info['reward'][key])
    for key in evaluation_names:
        observations[idx][key].append(info['evaluation'][key])
    #print('rewards', info['reward'])
    # observations[idx]['clothfunnel_default'].append(info['reward']['clothfunnel_default'])
    # observations[idx]['normalised_coverage'].append(info['reward']['normalised_coverage'])
    # observations[idx]['speedFolding_approx'].append(info['reward']['speedFolding_approx'])
    # observations[idx]['IoU'].append(info['reward']['IoU'])
    # observations[idx]['IoU_differance'].append(info['reward']['IoU_differance'])
    # observations[idx]['coverage_differance'].append(info['reward']['coverage_differance'])
    # observations[idx]['learningToUnfold_approx'].append(info['reward']['learningToUnfold_approx'])

def get_agent(
        agent_name, agent_config_name, 
        arena_name, agent_save_dir, checkpoint=-1):
    if 'oracle' not in agent_name:
        config = ag_ar.retrieve_config(
            agent_name, 
            arena_name, 
            agent_config_name)
    else:
        config = DotMap({
            'oracle': True
        })
    
    agent = ag_ar.build_agent(agent_name, config=config)
    save_dir = os.path.join(agent_save_dir, arena_name, agent_name, agent_config_name)
    agent.set_log_dir(save_dir)
    print('Finished building agent {}'.format(agent_name))
    if 'oracle' not in agent_name:
        if checkpoint != -1:
            agent.load_checkpoint(checkpoint)
        else:
            print('Loading the latest checkpoint')
            agent.load()
    return agent


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--parallel_processes', type=int, default=4)
    parser.add_argument('--oracle', type=str, default='RealAdapt-OTS') # OTS, RealAdapt OTS, masked-biased-random.
    parser.add_argument('--domain', type=str, default='realadapt-towels')
    parser.add_argument('--single_agent', action='store_true')
    parser.add_argument('--checkpoint', type=int, default='-1')
    return parser.parse_args()


def main():
    ray.init()

    arg_parser = argument_parser()
    arena_name = f'softgym|domain:{arg_parser.domain},initial:crumpled,action:pixel-pick-and-place(1),task:flattening,horizon:20'
    if arg_parser.oracle == 'RealAdapt-OTS':
        agent_name = 'oracle-rect-fabric|action:pixel-pick-and-place(1),strategy:realadapt,task:flattening'
    elif arg_parser.oracle == 'RealAdapt-OTS':
        agent_name = 'oracle-rect-fabric|action:pixel-pick-and-place(1),strategy:oracle-towel,task:flattening'
    elif arg_parser.oracle == 'masked-biased-random':
        agent_name = 'oracle-garment|mask-biased-pick-and-place'
    arenas = setup_arenas(arena_name, num_processes=arg_parser.parallel_processes)
    process = []
    for i, arena in enumerate(arenas):
        arena_id = arena._actor_id.hex()
        print(f"Actor {i} Ray ID: {arena_id}")
        process.append(arena.set_id.remote(arena_id))
        process.append(arena.set_train.remote())
    ray.get(process)
        # ray.get(arena.set_id.remote(arena_id))
        # ray.get(arena.set_train.remote())

    arena_ids = ray.get([e.get_id.remote() for e in arenas])
    
    if arg_parser.single_agent:
        print('Single Agent !!!')
        agent = get_agent(
            agent_name, 
            "default", 
            "",
            "",
            -1)
        agents = [agent for _ in range(len(arenas))]
        agent.reset(arena_ids)
    else:
        agents = [get_agent(
            agent_name, 
            "default", 
            "",
            "",
            -1) for _ in range(len(arenas))]
        for i, agent in enumerate(agents):
            agent.reset([arena_ids[i]])
        print('Multiple Agents !!!', len(agents))
    
    observations = [{
        k: [] for k in obs_config.keys()
    } for _ in range(len(arenas))]
    actions = [{'norm-pixel-pick-and-place': []} for _ in range(len(arenas))]

    data_path = "policy-{}-domain-{}-towels".format(arg_parser.oracle, arg_parser.domain)

    dataset = TrajectoryDataset(
        data_path=data_path,
        io_mode='a',
        obs_config=obs_config,
        act_config=action_config,
        whole_trajectory=True
    )
    
    waiting_infos = [e.reset.remote() for e in arenas]
    ready_arenas = []
    ready_actions = []
    ready_infos = []
    
    print('saved data', dataset.num_trajectories())
    ## intialise tdqm
    pbar = tqdm(total=arg_parser.trials, desc='Collecting Trajectories')
    pbar.update(dataset.num_trajectories())
    while dataset.num_trajectories() < arg_parser.trials:
        
        ready_actions = get_actions(agents, arena_ids, ready_infos, actions)
        
        ready_arenas, ready_infos, waiting_infos = step_arenas(
            arenas, arena_ids, 
            ready_arenas, ready_actions, 
            waiting_infos)
        
        new_ready_infos = []
        for info in ready_infos:
            arena_id = info['arena_id']
            idx = arena_ids.index(arena_id)

            update_observations(observations, idx, info)
            
            if info['done']:
                ## reset the agent and arena
                
                dataset.add_trajectory(
                    observations[idx],
                    actions[idx])
                pbar.update(1)
                # clean observations and actions
                observations[idx] = {
                    k: [] for k in obs_config.keys()
                }
                actions[idx] = {'norm-pixel-pick-and-place': []}
                
                ready_arenas.remove(arenas[idx])
                if arg_parser.single_agent:
                    agents[idx].reset([arena_id])
                else:
                    agents[idx] = get_agent(arg_parser.agent_name, 
                            arg_parser.agent_config_name, 
                            arg_parser.agent_trained_arena_name,
                            arg_parser.agent_save_dir,
                            arg_parser.checkpoint)
                    agents[idx].reset([arena_id])
                waiting_infos.append(arenas[idx].reset.remote())
            else:
                new_ready_infos.append(info)
        ready_infos = new_ready_infos

if __name__ == '__main__':
    main()