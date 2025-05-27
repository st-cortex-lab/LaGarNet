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
    parser.add_argument('--data_path', type=str, default='gc_longsleeve_flattening_offline_rl_diffusion_200k_compressed')
    args = parser.parse_args()


    
    dataset = TrajectoryDataset(
        data_path=args.data_path,
        data_dir='./datasets',
        whole_trajectory=True,
        io_mode='r',
        obs_config=compress_obs_config,
        act_config=action_config
    )
    
    

if __name__ == '__main__':
    main()