import os
## add the parent directory to the system path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import obs_config, obs_config_, compress_obs_config, action_config, reward_names, plot_results, evaluation_names
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
        'gc_longsleeve_flattening_offline_rl_diffusion_200k_compressed',
        'gc_longsleeve_flattening_offline_rl_mask_biased_200k_compressed',
    ]

    need_trj_nums = [
        10000,
        5000,
    ]

    target_datadir = f'gc_{args.obj}_flattening_offline_rl_mix_300k_2ratio1'

    if args.obj == 'mono-fabric':
        data_path = [
            'gc_msf_flattening_diffusion_50k',
            'gc_msf_flattening_masked_bias_50k',
            
            # 'gc_mono_fabric_flattening_offline_rl_mix_200k'
        ]

        target_datadir = 'gc_msf_flattening_offline_rl_mix_100k'

   

    datasets = []
    for path in data_path:
        dataset = TrajectoryDataset(
            data_path=path,
            data_dir='./datasets',
            whole_trajectory=True,
            io_mode='r',
            obs_config=compress_obs_config,
            act_config=action_config
        )
        datasets.append(dataset)

    target_dataset = TrajectoryDataset(
        data_path=target_datadir,
        data_dir='./datasets',
        whole_trajectory=True,
        io_mode='w',
        obs_config=compress_obs_config,
        act_config=action_config
    )
    for i, dataset in tdqm(enumerate(datasets), desc='Traversing datasets'):
        for j in tdqm(range(need_trj_nums[i]), desc='Adding data'):
            total_data += 1
            data = dataset[j]
            target_dataset.add_trajectory(data['observation'], data['action'])
        
    print('Total trj:', total_data)
    

if __name__ == '__main__':
    main()