# This script should be run under the conda envrionment of `ros2-humbel` branch of `agent-arena`
# python ur3e_control.py --agent_config RGB2M-on-mix-800k-all-garment-bonus-penalty-v2 --wm gc_rssm --goal_conditioned
import os
import traceback
import yaml
import json
import cv2
from dotmap import DotMap
from matplotlib import cm
import rclpy

import agent_arena as agar
from agent_arena import TrainableAgent

import argparse
from robot_control_cloth.interface.utils import *
from robot_control_cloth.interface.agent_arena_interface import AgentArenaInterface

from lagarnet.controllers.workspace_sampler_planet_clothpick \
    import WorkspaceSamplerPlaNetClothPick
from lagarnet.controllers.planet.rssm import RSSM

from agent_arena.utilities.visual_utils import \
    draw_pick_and_place, plot_image_trajectory
from lagarnet.controllers.gc_rssm import GC_RSSM
from lagarnet.controllers.mask_biased_pixel_pick_and_place_policy \
    import GarmentMaskBiasedPixelPickAndPlacePolicy


def success_check_fn(state):
    return state['normalised_coverage'] > 0.9 and state['max_IoU'] > 0.8

def visualise_action_inference(internal_states, save_dir=None):

    if len(internal_states) == 0:
        return

    print('internal_states', internal_states[0].keys())

    interation_means = [its['iteration_means'] for its in internal_states] # T * I * 1 * 4
    interation_means = np.stack(interation_means, axis=0)
    print('iteration_means', interation_means.shape)
    input_obs = [its['input_obs'] for its in internal_states]
    input_obs = (np.stack(input_obs, axis=0)[:, :, :, :3].clip(0, 1) * 255).astype(np.uint8)
    print('input_obs', input_obs.shape)

    recon_obs = [its['recon_obs'] for its in internal_states]
    recon_obs = (np.stack(input_obs, axis=0)[:, :, :, :3].clip(0, 1) * 255).astype(np.uint8)
    print('recon_obs', recon_obs.shape)
    if recon_obs.shape[-1] == 1:
        recon_obs = np.repeat(recon_obs, 3, axis=-1)
        print(f'Converted grayscale to RGB: {recon_obs.shape}')



    posterior_rewards = [its['posterior_reward'] for its in internal_states] # T * I
    posterior_rewards = np.stack(posterior_rewards, axis=0)
    print('posterior_rewards', posterior_rewards)

    prior_rewards = []

    # for each T in interation_means, choose 10 samples evenly from the second dimension including the first and last samples   
    T, I, _, _ = interation_means.shape
    S = 20
    colormap = cm.get_cmap('cool') #YlOrRd
    chosen_means = np.zeros((interation_means.shape[0], S, 4))
    for i in range(interation_means.shape[0]):
        for j in range(S-1):
            chosen_means[i, j] = interation_means[i, j * I // S]
        chosen_means[i, -1] = interation_means[i, -1]

    draw_action_img = []
    # start_color = np.array([52, 255, 0])
    # end_color = np.array([139, 0, 0])
    for i in range(T):
        img = input_obs[i]
        ## resize image to 256 by 256
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        #img = (input_obs[i] * 255).clip(0, 255).astype(np.uint8)
        H, W, C = img.shape
        for j in range(S):
            pick = (chosen_means[i, j, :2] + 1)/2 * np.asarray([H, W])
            place = (chosen_means[i, j, 2:] + 1)/2 * np.asarray([H, W])
            
            alpha = j / (S - 1)
            color = colormap(alpha)
            color = np.array(color[:3]) * 255

            img = draw_pick_and_place(
                img, pick, place, get_ready=True, color=color)
        draw_action_img.append(img)
    
    plot_image_trajectory(
        draw_action_img, 
        save_path=f'{save_dir}/visualise_inference', 
        title='iteration')

    last_steps_costs = [its['last_costs'] for its in internal_states]
    last_steps_samples = [its['last_samples'] for its in internal_states]
    chosen_samples_list = []
    chosen_costs_list = []
    for last_costs, last_samples in zip(last_steps_costs, last_steps_samples):
        # sort last_samples by last_costs
        last_samples = last_samples[np.argsort(last_costs)]
        # sort last_costs
        last_costs = np.sort(last_costs)
        print('last_samples', last_samples.shape)
        print('last_costs', last_costs.shape)
        # evenly choose S samples
        chosen_samples = last_samples[::len(last_samples)//S, :].reshape(-1, 4)
        #print('chosen_samples 1', chosen_samples.shape)
        # add the last sample
        #chosen_samples = np.append(chosen_samples, last_samples[-1])
        chosen_costs = -last_costs[::len(last_costs)//S]
        print('chosen costs')
        prior_rewards.append(chosen_costs[0])
        chosen_samples_list.append(chosen_samples[::-1])
        chosen_costs_list.append(chosen_costs[::-1])
        #chosen_costs = np.append(chosen_costs, last_costs[-1])
        print('chosen_samples', chosen_samples.shape)
        print('chosen_costs', chosen_costs.shape)

    high_reward_color = np.array([52, 255, 0]) # when reward is 0.5
    low_reward_color = np.array([139, 0, 0]) # when reward is -0.5
    colormap = cm.get_cmap('autumn') #YlOrRd

    low_reward = 0
    high_reward = 0.5
    chosen_costs_list = [np.clip(cost, low_reward, high_reward) for cost in chosen_costs_list]
    
    draw_action_img = []
    for i in range(T):
        img = recon_obs[i]
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        H, W, C = img.shape
        for j in range(len(chosen_samples_list[i])):
            pick = (chosen_samples_list[i][j, :2] + 1)/2 * np.asarray([H, W])
            place = (chosen_samples_list[i][j, 2:] + 1)/2 * np.asarray([H, W])

            alpha = (chosen_costs_list[i][j] - low_reward) / (high_reward - low_reward)
            alpha = 1-alpha
            # color = low_reward_color + (high_reward_color - low_reward_color) * alpha
            color = colormap(alpha)
            color = np.array(color[:3]) * 255
            img = draw_pick_and_place(
                img, pick, place, get_ready=True, color=color)
        draw_action_img.append(img)
    
    plot_image_trajectory(draw_action_img, 
                          save_path=f'{save_dir}/visualise_inference', 
                          title='last_step_reward')

    pick_masks = [its['pick-mask'] for its in internal_states]
    place_masks = [its['place-mask'] for its in internal_states]

    plot_image_trajectory(pick_masks, save_path=f'{save_dir}/visualise_inference', title='pick_masks')
    plot_image_trajectory(place_masks, save_path=f'{save_dir}/visualise_inference', title='place_masks')

    # save estimated rewards in human-readable format
    posterior_rewards = np.array(posterior_rewards)
    prior_rewards = np.array(prior_rewards)
    json_data = {
        'posterior_rewards': posterior_rewards.tolist(),
        'prior_rewards': prior_rewards.tolist()
    }
    with open(f'{save_dir}/info.json', 'w') as f:
        json.dump(json_data, f, indent=4)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='flattening')
    parser.add_argument('--log_dir', default='/data/models')
    parser.add_argument('--save_dir', default='./real_results')
    #parser.add_argument('--store_interm', action='store_true', help='store intermediate results')
    parser.add_argument('--eval_checkpoint', default=-1, type=int)

    parser.add_argument('--config',  default='RGB2M-on-mix-200k-sf')
    parser.add_argument('--object', default='longsleeve', type=str)
    parser.add_argument('--sampler_config', default='cloth-mask-centre-multiple-prioritise-middle', type=str)

    ## Sim2Real Protocol
    parser.add_argument('--depth_sim2real', default='v2')
    parser.add_argument('--mask_sim2real', default='v2')
    parser.add_argument('--sim_camera_height', default=0.65, type=float)

    ## Grasping Protocol
    parser.add_argument('--disable_adj_pick', action='store_true')
    parser.add_argument('--disable_adj_orient', action='store_true')


    parser.add_argument('--goal_conditioned', action='store_true')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wm', default='rssm', type=str)




    return parser.parse_args()

if __name__ == "__main__":
        
    args = parse_arguments()

    max_steps = 20

    #agent_name = "rssm"
    trained_arena = f"softgym|domain:clothfunnels-realadapt-{args.object},task:flattening,horizon:30"
    # capitalise the first letter of object
    capitalised_obj = args.object[0].upper() + args.object[1:]
    #test_arena = f"softgym|domain:ur3e-small{capitalised_obj},task:flattening,horizon:30"
    
    adjust_pick= not args.disable_adj_pick
    adjust_orien= not args.disable_adj_orient

    if args.wm in ['rssm', 'gc_rssm']:
        config = agar.retrieve_config(
            args.wm, 
            trained_arena, 
            args.config,
            config_dir='configuration')
        
        # TODO: Check if this is necessary
        config.transform.params.depth_clip_min = 0 #0.72
        config.transform.params.depth_clip_max = 1.0 #0.76
        config.transform.params.depth_min = 0 # 0.72
        config.transform.params.depth_max = 1.0 # 0.76
        config.transform.params.depth_eval_process = False
        config.transform.params.rgb_eval_process = False

        config.debug = args.debug
    

    #agent = agar.build_agent(agent_name, config)
    
    if args.wm == 'rssm':
        agent = RSSM(config)
        agent.set_eval()
    elif args.wm == 'gc_rssm':
        agent = GC_RSSM(config)
        agent.set_eval()
    elif args.wm == 'mask-biased-random':
        agent = GarmentMaskBiasedPixelPickAndPlacePolicy(DotMap())
        args.config = 'default'
    else:
        raise ValueError(f"Unknown world model: {args.wm}")
    
    #logger = ag_ar.build_logger(arena.logger_name, config.save_dir)
    save_dir = os.path.join(
        args.log_dir, f'ws_smplr_{args.wm}', f'{args.config}+{args.sampler_config}')
    model_dir = os.path.join(args.log_dir, trained_arena, f'{args.wm}', f'{args.config}')
    agent.set_log_dir(model_dir)

    config_file = os.path.join('configs', args.sampler_config + '.yaml')
   
    with open(config_file, 'r') as f:
        worspace_sampler_config = yaml.safe_load(f)

    worspace_sampler_config['agent'] = agent
    

    if args.wm in ['rssm', 'gc_rssm']:
        worspace_sampler_config['checkpoint'] = int(args.eval_checkpoint)
        worspace_sampler_config['policy'] = {'params': config.policy.params}


    worspace_sampler_config = DotMap(worspace_sampler_config)

    worspace_sampler_config.padding = 110 #220 #110
    worspace_sampler_config.debug = args.debug
    

    workspace_sampler = WorkspaceSamplerPlaNetClothPick(
        worspace_sampler_config)
    workspace_sampler.set_log_dir(save_dir)

    checkpoint = -1

    if isinstance(agent, TrainableAgent):
        if args.eval_checkpoint == -1:
            checkpoint = agent.load()
        else:
            agent.load_checkpoint(int(args.eval_checkpoint))
            checkpoint = args.eval_checkpoint
    config_name = args.wm + '+' + args.sampler_config + '+' + args.config
    print('Finsh Initialising Agent ...')

    sim2real = None
    try:
        rclpy.init()
        
        ### Run Sim2Real ###
        sim2real = AgentArenaInterface(
            workspace_sampler,config_name, 
            checkpoint, 
            task=args.task, 
            adjust_orien=adjust_orien, 
            adjust_pick=adjust_pick,
            steps=20,
            save_dir=args.save_dir,
            depth_sim2real=args.depth_sim2real,
            mask_sim2real=args.mask_sim2real,
            sim_camera_height=args.sim_camera_height,
            goal_conditioned=args.goal_conditioned,
            human_success_judgement=False,
            whole_workspace=True,
            success_check_fn = success_check_fn,
            callback_on_internal_states= (visualise_action_inference if isinstance(agent, TrainableAgent) else lambda x: None),
            )
        
        sim2real.run()
    except Exception as e:
        print(f'Caught exception: {e}')
        print('Stack trace:')
        traceback.print_exc()
        if sim2real is not None:
            sim2real.stop_video()
            rclpy.try_shutdown()
            sim2real.destroy_node()