# Plot Input Trajectory of a Trial

import argparse

import actoris_harena.api as ag_ar
from ..controllers.workspace_sampler_planet_clothpick \
    import WorkspaceSamplerPlaNetClothPick
from ..controllers.planet.rssm import RSSM


from dotmap import DotMap
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from actoris_harena.utilities.perform_single import perform_single
from actoris_harena.utilities.visual_utils import plot_image_trajectory

def save_depth_distribution(depth_image, filename='depth_distribution', directory="."):
    """
    Saves a histogram of depth values from a depth image to a PNG file.

    Parameters:
        depth_image (numpy.ndarray): A 2D array representing the depth image.
        filename (str): The name of the output file (without extension). Default is 'depth_distribution'.
        directory (str): The directory where the file will be saved. Default is the current directory (".").
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Compute histogram with 100 bins
    depth_image = depth_image.flatten()
    hist, bins = np.histogram(depth_image, bins=100)
    print('hist', hist)
    print('bins', bins)
    
    # Compute bin centers for better alignment in the bar plot
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot and save histogram
    plt.figure()
    plt.bar(bin_centers, hist, width=np.diff(bins), align='center')
    plt.xlabel('Depth Value')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.title('Depth Distribution')
    plt.savefig(f'{directory}/{filename}.png')
    plt.close()

def main():

    ### Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_config',  default='RGBD2M-on-mix-200k-sf')
    parser.add_argument('--log_dir', default='/data/planet-clothpick++')
    parser.add_argument('--verbose', default='silence', type=str)
    parser.add_argument('--disp', default=0, type=int)
    parser.add_argument('--eval_checkpoint', default='-1', type=int)
    parser.add_argument('--object', default='longsleeve', type=str)
    
    args = parser.parse_args()

    if args.disp == 1:
        disp = True
    elif args.disp == 0:
        disp = False
    
    agent_name = "planet-clothpick++"
    trained_arena = f"softgym|domain:clothfunnels-realadapt-{args.object},task:flattening,horizon:30"
    # capitalise the first letter of object
    capitalised_obj = args.object[0].upper() + args.object[1:]
    test_arena = trained_arena

   
    

    config = ag_ar.retrieve_config(
        agent_name, 
        trained_arena, 
        args.agent_config,
        config_dir='configuration')

    print(config)
    arena = ag_ar.build_arena(test_arena + ',disp:{}'.format(disp))
    agent = RSSM(config)
    #logger = ag_ar.build_logger(arena.logger_name, config.save_dir)
    save_dir = './results'
    model_dir = os.path.join(args.log_dir, trained_arena, f'{agent_name}', f'{args.agent_config}')
    arena.set_log_dir(save_dir)
    agent.set_log_dir(model_dir)
   
    eval_config = arena.get_eval_configs()
    eid = 0

    res = perform_single(
        arena,
        agent,
        mode='eval',
        episode_config=eval_config[eid],
        max_steps=30)
    
    import numpy as np
    input_obs = np.stack([iss['input_obs'] for iss in res['internal_states']])
    
    print('input obs shape', input_obs.shape)
    input_depth_obs = input_obs[:, :, :, -2]
    print('input dpeth max', input_depth_obs.max())
    print('input dpeth min', input_depth_obs.min())
    depth_list = []
    for i, depth in enumerate(input_depth_obs):
        d = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
        d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        depth_list.append(d)
        save_depth_distribution(depth, filename=f"input_depth_distribution_{i}", directory=save_dir)
    depth_list = np.stack(depth_list)

    plot_image_trajectory(depth_list, save_path=save_dir, title='input_depth_obs', col=5)
    
    # input_rgb_obs = (input_obs[:, :, :, :3].clip(0, 1) * 255).astype(np.uint8)
    # plot_image_trajectory(input_rgb_obs, save_path=save_dir, title='input_rgb_obs', col=5)

    depth_obs = np.stack([iss['observation']['depth'] for iss in res['information']])
    

    #print(res['internal_states'])
    print('depth obs shape', depth_list.shape)
    plot_image_trajectory(depth_obs, save_path=save_dir, title='depth', col=5)

    rgb_obs = np.stack([iss['observation']['rgb'] for iss in res['information']])
    print('rgb obs shape', rgb_obs.shape)
    plot_image_trajectory(rgb_obs, save_path=save_dir, title='rgb', col=5)

    mask_obs = np.stack([iss['observation']['mask'] for iss in res['information']])
    print('mask obs shape', mask_obs.shape)
    plot_image_trajectory(mask_obs, save_path=save_dir, title='mask', col=5)

if __name__ == '__main__':
    main()