# Plot latent reconstruction of trained model.

# Usage:
# python visualise_latent_reconstruction.py --config RGB2RGB-on-all-garment-800k \
# --data_path gc_dress_flattening_offline_rl_mix_200k_compressed  \
# --model_dir ../test_results

import argparse
import cv2
import os
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

import actoris_harena.api as ag_ar
from actoris_harena.utilities.visual_utils import plot_image_trajectory, plot_pick_and_place_trajectory
from actoris_harena.utilities.trajectory_dataset \
    import TrajectoryDataset
from actoris_harena.agent.utilities.torch_utils import *


from lagarnet.utils import obs_config, compress_obs_config, reward_names, plot_results, evaluation_names
from lagarnet.controllers.planet.rssm import RSSM

action_config = {
    'norm-pixel-pick-and-place': {'shape': (2, 2), 'output_key': 'default'},
}

# TODO: put this into agent-arena.
def print_information(data, indent=2):
    """
    Recursively prints the structure and details of the data in a tree-like format.
    :param data: The data structure to be printed (dict, list, np.ndarray, etc.)
    :param indent: The current level of indentation for tree structure
    """
    # Create an indentation string based on the current level
    prefix = " " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}{key}:")
            print_information(value, indent + 4)  # Increase indentation for nested structures
    elif isinstance(data, list):
        print(f"{prefix}List of {len(data)} items:")
        print_information(data[0], indent + 4)
    elif isinstance(data, np.ndarray):
        print(f"{prefix}Numpy array with shape {data.shape}, type {data.dtype}")
    elif isinstance(data, torch.Tensor):
        print(f"{prefix}PyTorch tensor with shape {data.shape}, type {data.type}")
    else:
        print(f"{prefix}scalar with type {type(data)}")

# Initialize LPIPS model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = LPIPS(net='squeeze', normalise=True).to(device)
kid = KernelInceptionDistance(normalize=True, subset_size=5).to(device)
fid = FrechetInceptionDistance(feature=64, normalize=True).to(device)

# Lower LPIPS indicates higher similarity (better perceptual quality).

def mse_score(x, y):
    """
    Calculate the MSE score between two images, ranged between 0 and 1.
    :param x: First image (numpy array)
    :param y: Second image (numpy array)
    :return: MSE score
    """
    # Convert images to PyTorch tensors and move to GPU if available
    # x = torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    # y = torch.from_numpy(y).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

    # Calculate MSE score
    return mse(x, y) #.item()

def lipips_score(x, y):
    """
    Calculate the LPIPS score between two images, ranged between 0 and 1.
    :param x: First image (numpy array)
    :param y: Second image (numpy array)
    :return: LPIPS score
    """
    # Convert images to PyTorch tensors and move to GPU if available
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device)
    y = torch.from_numpy(y).permute(0, 3, 1, 2).float().to(device)
    # Normalize images to the range [-1, 1]
    x = x * 2 - 1
    y = y * 2 - 1

    # Calculate LPIPS score
    return lpips_model(x, y).item()

def ssim_score(x, y):
    """
    Calculate the SSIM score between two images, ranged between 0 and 1.
    :param x: First image (numpy array)
    :param y: Second image (numpy array)
    :return: SSIM score
    """
    # Convert images to PyTorch tensors and move to GPU if available
    # x = torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    # y = torch.from_numpy(y).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

    # Calculate SSIM score
    return ssim(x, y, data_range=1.0, channel_axis=-1)
    

def kid_score(x, y):
    """
    Calculate the KID score between two images, input ranged between 0 and 1.
    :param x: First image (numpy array)
    :param y: Second image (numpy array)
    :return: KID score
    """
    # Convert images to PyTorch tensors and move to GPU if available
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device)
    y = torch.from_numpy(y).permute(0, 3, 1, 2).float().to(device)

    # Calculate KID score
    kid.update(x, real=True)
    kid.update(y, real=False)
    score = kid.compute()
    kid.reset()
    #print('KID score:', score)

    return score[0].item()

def fid_score(x, y):
    """
    Calculate the FID score between two images, input ranged between 0 and 1.
    :param x: First image (numpy array)
    :param y: Second image (numpy array)
    :return: FID score
    """
    # Convert images to PyTorch tensors and move to GPU if available
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device)
    y = torch.from_numpy(y).permute(0, 3, 1, 2).float().to(device)

    # Calculate FID score
    fid.update(x, real=True)
    fid.update(y, real=False)
    score = fid.compute()

    return score.item()

def get_recon_results(agent, recon_object, config_name, 
                      obs_config, act_config, world_model,
                      sample_mode='eval', posterior_horizon=3, prior_horizon=3, 
                      recon_format='rgb', trj_nums=10):
    data_path = f'gc_{recon_object}_flattening_offline_rl_mix_200k_compressed'
    
    dataset = TrajectoryDataset(
        data_path=data_path,
        whole_trajectory=True,
        io_mode='r',
        sample_mode=sample_mode,
        obs_config=obs_config,
        act_config=act_config,
        data_dir='../data/datasets'
    )
    ground_truths = []
    recon_images = []


    metrics = {
        'mse': lambda x, y: mse_score(x, y), # Lower MSE indicates better quality
        'ssim': lambda x, y: ssim_score(x, y), # Higher SSIM indicates better quality
        'lpips': lambda x, y: lipips_score(x, y), # Lower LPIPS indicates better quality
        'fid': lambda x, y: fid_score(x, y), # Lower FID indicates better quality
        'kid': lambda x, y: kid_score(x, y), # Lower KID indicates better quality
    }

    for i in range(trj_nums):
        data = dataset[i]
        #print(data.keys())
        
        observation_list = [{k: v[i] for k, v in data['observation'].items()} for i in range(10)]

        ### Get Initial Latent State
        init_info = {
            'arena_id': 0,
            'observation': observation_list[0]
        }
        agent.init([init_info])
        init_latent_state = agent.get_state()[0]['latent_state']

        ### Get Posterior Latent States
        process_data = agent._preprocess(data, train=False, single=True)
        #print(process_data['input_obs'].shape)     

        input_state = init_latent_state
        input_state['input_obs'] = process_data['input_obs'][1:posterior_horizon+1]
        if world_model == 'gc_rssm':
            input_state['goal_obs'] = process_data['goal_obs'][1:posterior_horizon+1]

        # print('\n!!!!! Input')
        # print_information(input_state)

        posterior_latent_states, last_post = agent.unroll_state_action(
            input_state,
            process_data['action'][:posterior_horizon]
        )

        # print('\n!!!!! Posterior Output')
        # print_information(posterior_latent_states)
        
        posterior_recon_image = agent.visual_reconstruct(posterior_latent_states)

        # print('\n!!!!! Posterior Recon')
        # print_information(posterior_recon_image) # T, B, C, H, W

        ### Get Piror Latent States
        if world_model == 'gc_rssm':
            prior_latent_states = agent.unroll_action(
                last_post,
                process_data['action'][posterior_horizon: posterior_horizon + prior_horizon],
                goal_obs=process_data['goal_obs'][posterior_horizon: posterior_horizon + prior_horizon]
            )
        elif world_model == 'rssm':
            prior_latent_states = agent.unroll_action(
                last_post,
                process_data['action'][posterior_horizon: posterior_horizon + prior_horizon]
            )
        else:
            raise ValueError(f"Unknown world model: {world_model}")

        # print('\n!!!!! Prior Output')
        # print_information(prior_lastent_states)

        prior_recon_image = agent.visual_reconstruct(prior_latent_states)

        # print('\n!!!!! Prior Recon')
        # print_information(prior_recon_image)
        filename = ''
        if recon_format == 'rgb':
            posterior_recon_image = (posterior_recon_image.clip(0, 1)*255)
        plot_image_trajectory(
            posterior_recon_image[:, 0], 
            save_path='tmp/recon/{}/{}/{}/trial_{}'.format(world_model, config_name, recon_object, i),
            title='posterior'
        )

        if recon_format == 'rgb':
            prior_recon_image = (prior_recon_image.clip(0, 1)*255)

        plot_image_trajectory(
            prior_recon_image[:, 0], 
            save_path='tmp/recon/{}/{}/{}/trial_{}'.format(world_model, config_name, recon_object, i),
            title='prior'
        )

        gt = ts_to_np(process_data['output_obs'][1: posterior_horizon + prior_horizon + 1])\
            .transpose(0, 1, 3, 4, 2)
        if recon_format == 'rgb':
            gt = (gt*255)
        print('gt shape', gt.shape)

        actions = ts_to_np(process_data['action'][1:posterior_horizon + prior_horizon+1])


        plot_pick_and_place_trajectory(
            gt[:, 0],
            actions,
            save_path='tmp/recon/{}/{}/{}/trial_{}'.format(world_model, config_name, recon_object, i),
            title='ground_truth'
        )

        recon_image = np.concatenate((posterior_recon_image, prior_recon_image), axis=0)
        gt = (gt[:, 0]/255).clip(0, 1)
        recon_image = (recon_image[:, 0]/255).clip(0, 1)

        ground_truths.append(gt)
        recon_images.append(recon_image)

    ground_truths = np.stack(ground_truths)
    recon_images = np.stack(recon_images)
    
    # Ground turth to ground truth
    gt_scores = {k: m(ground_truths.reshape(-1, 64, 64, 3), ground_truths.reshape(-1, 64, 64, 3))
                for k, m in metrics.items()}
    print('\n\n')
    print('Trajectory Number:', trj_nums)
    print('Ground Truth Evaluation on {} steps'.format(posterior_horizon + prior_horizon))
    for k, v in gt_scores.items():
        print('{}: {:.4f}'.format(k, v))

    results = {}

    posterior_scores = {k: m(ground_truths[:, :posterior_horizon, :, :, :3].reshape(-1, 64, 64, 3), recon_images[:, :posterior_horizon, :, :, :3].reshape(-1, 64, 64, 3))
                        for k, m in metrics.items()}
    
    results['posterior'] = posterior_scores

    print('\n\n')
    print('Trajectory Number:', trj_nums)
    print('Posterior Evaluation on {} steps'.format(posterior_horizon))
    for k, v in posterior_scores.items():
        print('{}: {:.4f}'.format(k, v))

    for i in range(prior_horizon):
        prior_scores = {
            k: m(ground_truths[:, posterior_horizon + i, :, :, :3], recon_images[:, posterior_horizon + i, :, :, :3])
            for k, m in metrics.items()
        }
        print('\n\n')
        print('{}-Step Prior Evaluation'.format(i+1))
        
        for k, v in prior_scores.items():
            print('{}: {:.4f}'.format(k, v))

        results['prior_{}'.format(i+1)] = prior_scores
    
    return results

def main():

    ### Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  default='RGB2RGB-on-mix-200k-sf')
    parser.add_argument('--model_dir', default='../test_results/models', type=str)
    parser.add_argument('--verbose', default='silence', type=str)
    parser.add_argument('--disp', default=0, type=int)
    parser.add_argument('--eval_checkpoint', default='-1', type=int)
    parser.add_argument('--arena_object', default='longsleeve', type=str)
    parser.add_argument('--wm', default='rssm', type=str)
    #parser.add_argument('--recon_object', default='longsleeve', type=str)
    parser.add_argument('--data_path', type=str, default='gc_longsleeve_flattening_offline_rl_mix_200k_compressed')
    parser.add_argument('--posterior_horizon', default=3)
    parser.add_argument('--prior_horizon', default=3)
    parser.add_argument('--sample_mode', default='eval')
    parser.add_argument('--recon_format', default='rgb', type=str)
    parser.add_argument('--trj_nums', type=int, default=10)

    #data_path = f'gc_{args.recon_object}_flattening_offline_rl_mix_200k_compressed'

    
    args = parser.parse_args()

    if args.disp == 1:
        disp = True
    elif args.disp == 0:
        disp = False
    
    trained_arena = f"softgym|domain:clothfunnels-realadapt-{args.arena_object},task:flattening,horizon:30"
    

    ## Load agent for latent visuslisation
    config = ag_ar.retrieve_config(
        args.wm, 
        trained_arena, 
        args.config,
        config_dir='../configuration')

    if args.wm == 'rssm':
        agent = RSSM(config)
    elif args.wm == 'gc_rssm':
        from lagarnet.controllers.gc_rssm import GC_RSSM
        agent = GC_RSSM(config)
    else:
        raise ValueError(f"Unknown world model: {args.wm}")
    
    
    model_dir = os.path.join(args.model_dir, args.wm, trained_arena, args.config)
    agent.load(model_dir)

    ## Load dataset
    obs_config = config.datasets[0].params.obs_config.toDict()
    act_config = config.datasets[0].params.act_config.toDict()
    print('obs config', obs_config)

    result_scores = {}

    for recon_obj in ['longsleeve', 'dress', 'skirt', 'pants']:

        result = get_recon_results(
            agent, 
            recon_obj, 
            args.config,
            obs_config, 
            act_config,
            args.wm,
            sample_mode=args.sample_mode,
            posterior_horizon=args.posterior_horizon,
            prior_horizon=args.prior_horizon,
            recon_format=args.recon_format,
            trj_nums=args.trj_nums
        )
        result_scores[recon_obj] = result

        
        
    # save the above printing to a csv file named with config name, first columns is posterior or prior, second columns is the object type, and other columns are the metrics

    cvs_file = 'tmp/recon/{}/{}/score.csv'.format(args.wm, args.config)
    # if not os.path.exists(os.path.dirname('tmp/recon/metrics')):
    #     os.makedirs(os.path.dirname('tmp/recon/metrics'))

    with open(cvs_file, 'w') as f:
        f.write('Posterior/Prior, Object Type, MSE, SSIM, LPIPS, FID, KID\n')
        for recon_obj in result_scores.keys():
            for k, v in result_scores[recon_obj].items():
                if k == 'posterior':
                    f.write('Posterior, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(recon_obj, v['mse'], v['ssim'], v['lpips'], v['fid'], v['kid']))
                else:
                    f.write('Prior {}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(k.split('_')[1], recon_obj, v['mse'], v['ssim'], v['lpips'], v['fid'], v['kid']))
    print('Results saved to {}'.format(cvs_file))

    
    
if __name__ == '__main__':
    main()