from planet_clothpick_plus_plus.utils import obs_config, action_config, compress_obs_config, reward_names, plot_results, evaluation_names
from agent_arena.utilities.trajectory_dataset \
    import TrajectoryDataset
import argparse
from agent_arena.utilities.transform.pick_and_place_transformer_v1 import PickAndPlaceTransformerV1
from tqdm import tqdm
from dotmap import DotMap
ts_to_np = lambda x: x.cpu().detach().numpy()

# transform:
#   name: 'pick_and_place_transformer'
#   params:
#     img_dim: [64, 64]
#     remap_image: [0, 1]
#     device: "cuda:0"
#     random_rotation: True
#     random_terminal: False
#     rotation_degree: 1
#     all_goal_rotate: True
#     goal_translate: 0.2
#     vertical_flip: True
#     random_resize: False
#     action_noise: 0.0
#     depth_clip: True
#     depth_clip_min: 1.8 #
#     depth_clip_max: 2.0 #0.66
#     min_max_norm: True
#     depth_min: 1.8
#     depth_max: 2.0
#     depth_map: True
#     depth_map_range: [0, 1]
#     depth_blur: True
#     depth_flip: True
#     depth_blur_kernel_size: 2
#     depth_noise_var: 0.01
#     depth_eval_process: True
#     maskout: True
#     bg_value: 0

def argument_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--parallel_processes', type=int, default=4)
    # parser.add_argument('--agent_name', type=str, default='oracle-garment|mask-biased-pick-and-place')
    # parser.add_argument('--arena_name', type=str, default='softgym|domain:clothfunnels-real2sim-longsleeve,task:flattening,horizon:20')
    parser.add_argument(
        '--path', type=str, 
        default='planet-clothpick++_data/gc_longsleeve_flattening_offline_rl_diffusion_200k')

    return parser.parse_args()


def main():
    args = argument_parser()
    data_path = args.path
    dataset = TrajectoryDataset(
        data_path=data_path,
        whole_trajectory=True,
        io_mode='r',
        obs_config=compress_obs_config,
        act_config=action_config
    )

    config = {
        'img_dim': [64, 64],
        'rgb_norm_mode': ['remap'],
        'rgb_norm_param': {'remap_range': [0, 1]},
        'rgb_noise_factor': 0.01,
        'device': 'cpu',
        'random_rotation': True,
        'random_terminal': False,
        'rotation_degree': 1,
        'all_goal_rotate': True,
        'goal_translate': 0.2,
        'preserve_goal_mask': False,
        'vertical_flip': True,
        'random_resize': False,
        'action_noise': 0.0,
        'depth_clip': True,
        'depth_clip_min': 1.8,
        'depth_clip_max': 2.0,
        'min_max_norm': True,
        'depth_min': 1.8,
        'depth_max': 2.0,
        'depth_map': True,
        'depth_map_range': [0, 1],
        'depth_blur': True,
        'depth_flip': True,
        'depth_blur_kernel_size': 2,
        'depth_noise_var': 0.01,
        'depth_eval_process': True,
        'maskout': True,
        'bg_value': 0
    }
    config = DotMap(config)
    transform = PickAndPlaceTransformerV1(config)

    sequence = 3
    
    dataset.set_transform(transform)

    for i in tqdm(range(0, 5)):
        data = dataset[i]
        rgbs = (ts_to_np(data['rgb']).transpose(0, 2, 3, 1) * 255).clip(0, 255).astype('uint8')
        goal_rgbs = (ts_to_np(data['goal-rgb']).transpose(0, 2, 3, 1)*255).clip(0, 255).astype('uint8')
        # print('data keys', data['observation'].keys(), data['action'].shape)
        actions = ts_to_np(data['action'])
        # reward_dict = {key: data['observation'][key] for key in reward_names}
        # evaluation_dict = {key: data['observation'][key] for key in evaluation_names}

        depths = ts_to_np(data['depth']).transpose(0, 2, 3, 1)
        goal_depths = ts_to_np(data['goal-depth']).transpose(0, 2, 3, 1)

        masks = ts_to_np(data['mask']).transpose(0, 2, 3, 1)
        goal_masks = ts_to_np(data['goal-mask']).transpose(0, 2, 3, 1)

        for k, v in data.items():
            print(k, v.shape)
        
        plot_results(
            rgbs, depths, masks,
            goal_rgbs, goal_depths, goal_masks,
            actions, filename='transform_data_{}'.format(i), sequence=sequence)
        
    dataset.set_transform(None)
    for i in tqdm(range(0, 5)):
        data = dataset[i]
        data_obs = data['observation']
        rgbs = data_obs['rgb'] #.transpose(0, 2, 3, 1)
        goal_rgbs = data_obs['goal-rgb'] #.transpose(0, 2, 3, 1)
        # print('data keys', data['observation'].keys(), data['action'].shape)
        actions = data['action']['norm-pixel-pick-and-place']
        # reward_dict = {key: data['observation'][key] for key in reward_names}
        # evaluation_dict = {key: data['observation'][key] for key in evaluation_names}

        depths = data_obs['depth'] #.transpose(0, 2, 3, 1)
        goal_depths = data_obs['goal-depth'] #.transpose(0, 2, 3, 1)

        masks = data_obs['mask'] #.transpose(0, 2, 3, 1)
        goal_masks = data_obs['goal-mask'] #.transpose(0, 2, 3, 1)
        plot_results(
            rgbs, depths, masks,
            goal_rgbs, goal_depths, goal_masks,
            actions, filename='un_transform_data_{}'.format(i), sequence=sequence)
    

if __name__ == '__main__':
    main()