from planet_clothpick_plus_plus.utils import obs_config, compress_obs_config, action_config, reward_names, plot_results, evaluation_names
from actoris_harena.utilities.trajectory_dataset \
    import TrajectoryDataset
import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='gc_all_garment_flattening_offline_rl_mix_800k_compressed')

    return parser.parse_args()

def main():
    args = argument_parser()
    data_path = args.path
    dataset = TrajectoryDataset(
        data_path=data_path,
        whole_trajectory=True,
        io_mode='r',
        obs_config=compress_obs_config,
        act_config=action_config,
        data_dir='../data'
    )

    for i in range(9, 11):
        data = dataset[i]
        rgbs = data['observation']['rgb']
        # print('data keys', data['observation'].keys(), data['action'].shape)
        actions = data['action']['norm-pixel-pick-and-place']
        reward_dict = {key: data['observation'][key] for key in reward_names}
        evaluation_dict = {key: data['observation'][key] for key in evaluation_names}

        depths = data['observation']['depth'] - 1.5
        masks = data['observation']['mask']

        print('rgb shape', rgbs.shape)
        
        plot_results(
            rgbs, depths, masks,    
            actions=actions, reward_dict=reward_dict, 
            evaluation_dict=evaluation_dict, filename='data_{}'.format(i))
    
    

if __name__ == '__main__':
    main()