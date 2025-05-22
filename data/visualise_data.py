# Author: Halid Kadi

from utils import obs_config, action_config, reward_names, plot_results, evaluation_names #see from the folder
from trajectory_dataset import TrajectoryDataset ## see from the folder
import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel_processes', type=int, default=4)
    parser.add_argument('--agent_name', type=str, default='oracle-garment|mask-biased-pick-and-place')
    parser.add_argument('--arena_name', type=str, default='softgym|domain:clothfunnels-real2sim-longsleeve,task:flattening,horizon:20')

    return parser.parse_args()

def main():
    args = argument_parser()
    data_path = args.agent_name + '_on_' + args.arena_name
    dataset = TrajectoryDataset(
        data_path=data_path,
        whole_trajectory=True,
        io_mode='r',
        obs_config=obs_config,
        act_config=action_config
    )

    for i in range(5):
        data = dataset[i]
        rgbs = data['observation']['rgb']
        # print('data keys', data['observation'].keys(), data['action'].shape)
        actions = data['action']['norm-pixel-pick-and-place']
        reward_dict = {key: data['observation'][key] for key in reward_names}
        evaluation_dict = {key: data['observation'][key] for key in evaluation_names}

        depths = data['observation']['depth'] - 1.5
        masks = data['observation']['mask']
        
        plot_results(
            rgbs, depths, masks,    
            actions, reward_dict, evaluation_dict, filename='data_{}'.format(i))
    
    

if __name__ == '__main__':
    main()