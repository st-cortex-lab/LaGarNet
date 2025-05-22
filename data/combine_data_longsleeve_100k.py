from planet_clothpick_plus_plus.utils import obs_config, action_config, reward_names, plot_results, evaluation_names
from agent_arena.utilities.trajectory_dataset \
    import TrajectoryDataset
import argparse
from tqdm import tqdm as tdqm


def main():
    total_data = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', type=str, default='longsleeve')
    args = parser.parse_args()

    data_path = [
        'planet-clothpick++_data/gc_longsleeve_flattening_offline_rl_diffusion_200k',
        'planet-clothpick++_data/gc_longsleeve_flattening_offline_rl_mask_biased_200k',
    ]

    target_datadir = f'gc_{args.obj}_flattening_offline_rl_mix_100k'
   

    datasets = []
    for path in data_path:
        dataset = TrajectoryDataset(
            data_path=path,
            whole_trajectory=True,
            io_mode='r',
            obs_config=obs_config,
            act_config=action_config
        )
        datasets.append(dataset)

    target_dataset = TrajectoryDataset(
        data_path=target_datadir,
        whole_trajectory=True,
        io_mode='w',
        obs_config=obs_config,
        act_config=action_config
    )
    for dataset in tdqm(datasets, desc='Traversing datasets'):
        for i in tdqm(range(2500), desc='Adding data'):
            total_data += 1
            data = dataset[i]
            target_dataset.add_trajectory(data['observation'], data['action'])
        
    print('Total trj:', total_data)
    

if __name__ == '__main__':
    main()