import argparse
import agent_arena.api as ag_ar
from agent_arena.utilities.perform_single import perform_single
import matplotlib.pyplot as plt
import numpy as np
from planet_clothpick_plus_plus.vidualisations.utils import plot_results
from dotmap import DotMap

def main():

    ### Argument Parsing
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--arena', 
    #     default="softgym|domain:mono-square-fabric,initial:crumple,"\
    #         "action:pixel-pick-and-place(1),task:flattening")
    # parser.add_argument('--agent', 
    #     default="oracle-rect-fabric|action:pixel-pick-and-place(1),"
    #             "strategy:expert,task:flattening")
    # parser.add_argument('--config',  default='default')
    # parser.add_argument('--log_dir', default='tmp')
    # parser.add_argument('--verbose', default='silence', type=str)
    # parser.add_argument('--disp', default=0, type=int)
    # parser.add_argument('--eval_checkpoint', default='-1', type=int)
   
    arena_name = 'softgym|domain:cloth-funnel-real2sim-longsleeve,task:flattening,horizon:10'
    agent_name = 'oracle-garment|mask-biased-pick-and-place'


    arena = ag_ar.build_arena(arena_name + ',disp:False')
    agent = ag_ar.build_agent(agent_name, config=DotMap({
        'oracle': True
    }))
    #logger = ag_ar.build_logger(arena.logger_name, config.save_dir)

    res = perform_single(arena, agent, mode='eval', 
        episode_config=None, collect_frames=False)
    
    # infos = res['informations']
    # rewards = [info['reward'] for info in infos]
    # rgbs = [info['observation']['rgb'] for info in infos]
    # actions = res['actions']
    # actions = [actions['norm-pixel-pick-and-place'] for actions in actions]
    # actions = [np.stack([action['pick_0'], action['place_0']])
    #            for action in actions]

    # reward_keys = rewards[0].keys()

    # rewad_dict = {key: [] for key in reward_keys}
    # for reward in rewards:
    #     for key in reward_keys:
    #         rewad_dict[key].append(reward[key])


    # print('rewards', rewad_dict)
    # rgbs = np.stack(rgbs)
    # actions = np.stack(actions)

    # # Visualize the rewards
    # plot_results(rgbs, actions, rewad_dict, filename='human')
    
    

if __name__ == '__main__':
    main()