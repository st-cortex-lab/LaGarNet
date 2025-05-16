import argparse
import agent_arena.api as ag_ar
from agent_arena.utilities.perform_single import perform_single
import os
import numpy as np
from planet_clothpick_plus_plus.vidualisations.utils import obs_config, action_config, reward_names, \
    plot_results, evaluation_names
import json


def main():

    ### Argument Parsing
    parser = argparse.ArgumentParser()
    # parser.add_argument('--arena', 
    #     default="softgym|domain:mono-square-fabric,initial:crumple,"\
    #         "action:pixel-pick-and-place(1),task:flattening")
    # parser.add_argument('--agent', 
    #     default="oracle-rect-fabric|action:pixel-pick-and-place(1),"
    #             "strategy:expert,task:flattening")
    parser.add_argument('--config',  default='default')
    parser.add_argument('--log_dir', default='tmp')
    parser.add_argument('--verbose', default='silence', type=str)
    parser.add_argument('--disp', default=0, type=int)
    parser.add_argument('--eval_checkpoint', default='-1', type=int)
    
    ## add arguments --load_saved as boolean for load the saved data 
    parser.add_argument('--load_saved', action='store_true')
    args = parser.parse_args()

    if not args.load_saved:
   
        arena_name = 'softgym|domain:clothfunnels-real2sim-longsleeve,task:flattening,horizon:14'
        agent_name = 'human-pixel-pick-and-place-1'


        arena = ag_ar.build_arena(arena_name + ',disp:False')
        agent = ag_ar.build_agent(agent_name)
        arena.set_eval()
        #logger = ag_ar.build_logger(arena.logger_name, config.save_dir)

        ## get the episode config where arena is in success state
        episode_config = None
        res = perform_single(arena, agent, mode='eval', 
            episode_config=episode_config, collect_frames=False, end_success=False)
        
        infos = res['informations']
        rewards = [info['reward'] for info in infos]
        rgbs = [info['observation']['rgb'] for info in infos]
        depths = [info['observation']['depth'] for info in infos]
        masks = [info['observation']['mask'] for info in infos]
        actions = res['actions']
        actions = [actions['norm-pixel-pick-and-place'] for actions in actions]
        actions = [np.stack([action['pick_0'], action['place_0']])
                for action in actions]

        reward_keys = rewards[0].keys()

        rewad_dict = {key: [] for key in reward_keys}
        for reward in rewards:
            for key in reward_keys:
                rewad_dict[key].append(float(reward[key]))
        
        evaluations = [info['evaluation'] for info in infos]
        evaluation_dict = {key: [] for key in evaluation_names}
        for evaluation in evaluations:
            for key in evaluation_names:
                evaluation_dict[key].append(float(evaluation[key]))


        print('rewards', rewad_dict)
        print('evaluations', evaluation_dict)
        rgbs = np.stack(rgbs)
        depths = np.stack(depths)
        if len(depths.shape) == 3:
            depths = np.expand_dims(depths, axis=3)
        masks = np.stack(masks)
        if len(masks.shape) == 3:
            masks = np.expand_dims(masks, axis=3)
        actions = np.stack(actions)
    else:
        rgbs = np.load('human_tmp_data/rgbs.npy')
        depths = np.load('human_tmp_data/depths.npy')
        masks = np.load('human_tmp_data/masks.npy')
        actions = np.load('human_tmp_data/actions.npy')
        with open('human_tmp_data/rewards.json', 'r') as f:
            rewad_dict = json.load(f)
        with open('human_tmp_data/evaluations.json', 'r') as f:
            evaluation_dict = json.load(f)

    # Visualize the rewards
    plot_results(
        rgbs, depths, masks,
        actions, rewad_dict, evaluation_dict, 
        filename='human')

    # save these data
    save_dir = 'human_tmp_data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir + '/rgbs.npy', rgbs)
    np.save(save_dir + '/depths.npy', depths)
    np.save(save_dir + '/masks.npy', masks)
    np.save(save_dir + '/actions.npy', actions)
    
    # save json file for rewards and evaluations
    with open(save_dir + '/rewards.json', 'w') as f:
        json.dump(rewad_dict, f)
    with open(save_dir + '/evaluations.json', 'w') as f:
        json.dump(evaluation_dict, f)
    
    

if __name__ == '__main__':
    main()