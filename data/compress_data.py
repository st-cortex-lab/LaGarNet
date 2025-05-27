import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import obs_config, compress_obs_config, \
    action_config, reward_names, obs_config_, plot_results, evaluation_names
from agent_arena.utilities.trajectory_dataset \
    import TrajectoryDataset
import argparse
from tqdm import tqdm as tdqm
import numpy as np
import cv2



def compress_data(data):
    res = {
        'action': data['action'].copy(),
        'observation': data['observation'].copy()
    }
    for key in ['rgb', 'depth', 'mask', 'goal-rgb', 'goal-depth', 'goal-mask']:
        if key in data['observation']:
            image = data['observation'][key]
            #print('image shape before:', image.shape)
            if 'mask' in key:
                image = image.astype(float)
            T = image.shape[0]
            ## resize the image to 64x64
            image = np.stack([cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA) for im in image])
            
            if 'mask' in key:
                image = (image > 0.9).astype(float)
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=3)
            #print('image shape after:', image.shape)
            res['observation'][key] = image
            
    return res

def main():
    total_data = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='gc_longsleeve_flattening_diffusion')
    args = parser.parse_args()


    target_datadir = f'{args.data_path}_compressed'

   
    dataset = TrajectoryDataset(
        data_path=args.data_path,
        data_dir='./datasets',
        whole_trajectory=True,
        io_mode='r',
        obs_config=obs_config, #obs_config_,
        act_config=action_config
    )

    target_dataset = TrajectoryDataset(
        data_path=target_datadir,
        data_dir='./datasets',
        whole_trajectory=True,
        io_mode='w',
        obs_config=compress_obs_config,
        act_config=action_config
    )

    for i in tdqm(range(len(dataset)), desc='Adding data'):
        total_data += 1
        data = dataset[i]
        data = compress_data(data)
        target_dataset.add_trajectory(data['observation'], data['action'])
        
    print('Total trj:', total_data)
    

if __name__ == '__main__':
    main()