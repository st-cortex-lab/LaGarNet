# This script should be run under the conda envrionment of `ros2-humbel` branch of `agent-arena`
# python ur3e_control.py --agent_config RGB2M-on-mix-800k-all-garment-bonus-penalty-v2 --wm gc_rssm --goal_conditioned
import os
import traceback
import yaml
import json
import cv2

import rclpy

import agent_arena as agar

import argparse
from robot_control_cloth.interface.utils import *
from robot_control_cloth.interface.agent_arena_interface import AgentArenaInterface

from controllers.workspace_sampler_planet_clothpick \
    import WorkspaceSamplerPlaNetClothPick
from controllers.planet.rssm import RSSM
from matplotlib import cm
from agent_arena.utilities.visual_utils import \
    draw_pick_and_place, plot_image_trajectory
from controllers.gc_rssm import GC_RSSM

def visualise_action_inference(internal_states, save_dir=None):

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



    estimated_rewards = [its['estimated_reward'] for its in internal_states] # T * I
    estimated_rewards = np.stack(estimated_rewards, axis=0)
    print('estimated_rewards', estimated_rewards)

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
    estimated_rewards = np.array(estimated_rewards)
    json_data = {
        'estimated_rewards': estimated_rewards.tolist(),
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

    parser.add_argument('--agent_config',  default='RGB2M-on-mix-200k-sf')
    parser.add_argument('--object', default='longsleeve', type=str)
    parser.add_argument('--sampler_config', default='cloth-mask-centre-multiple-prioritise-middle', type=str)

    ## Sim2Real Protocol
    parser.add_argument('--depth_sim2real', default='v2')
    parser.add_argument('--mask_sim2real', default='v2')
    parser.add_argument('--sim_camera_height', default=0.65, type=float)

    ## Grasping Protocol
    parser.add_argument('--adjust_pick', action='store_true')
    parser.add_argument('--adjust_orient', action='store_true')

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
    

    config = agar.retrieve_config(
        args.wm, 
        trained_arena, 
        args.agent_config,
        config_dir='configuration')
    
    # TODO: Check if this is necessary
    config.transform.params.depth_clip_min = 0 #0.72
    config.transform.params.depth_clip_max = 1.0 #0.76
    config.transform.params.depth_min = 0 # 0.72
    config.transform.params.depth_max = 1.0 # 0.76
    config.transform.params.depth_eval_process = False
    config.transform.params.rgb_eval_process = False
    

    #agent = agar.build_agent(agent_name, config)
    config.debug = args.debug
    if args.wm == 'rssm':
        agent = RSSM(config)
    elif args.wm == 'gc_rssm':
        agent = GC_RSSM(config)
    else:
        raise ValueError(f"Unknown world model: {args.wm}")
    agent.set_eval()
    #logger = ag_ar.build_logger(arena.logger_name, config.save_dir)
    save_dir = os.path.join(
        args.log_dir, f'ws_smplr_{args.wm}', f'{args.agent_config}+{args.sampler_config}')
    model_dir = os.path.join(args.log_dir, trained_arena, f'{args.wm}', f'{args.agent_config}')
    agent.set_log_dir(model_dir)

    config_file = os.path.join('configs', args.sampler_config + '.yaml')
   
    with open(config_file, 'r') as f:
        worspace_sampler_config = yaml.safe_load(f)

    worspace_sampler_config['agent'] = agent
    worspace_sampler_config['checkpoint'] = int(args.eval_checkpoint)
    worspace_sampler_config['policy'] = {'params': config.policy.params}

    worspace_sampler_config = DotMap(worspace_sampler_config)

    worspace_sampler_config.padding = 110
    worspace_sampler_config.debug = args.debug
    

    workspace_sampler = WorkspaceSamplerPlaNetClothPick(
        worspace_sampler_config)
    workspace_sampler.set_log_dir(save_dir)


    if args.eval_checkpoint == -1:
        checkpoint = agent.load()
    else:
        agent.load_checkpoint(int(args.eval_checkpoint))
        checkpoint = args.eval_checkpoint
    config_name = args.sampler_config + '+' + args.agent_config
    print('Finsh Initialising Agent ...')

    sim2real = None
    try:
        rclpy.init()
        
        ### Run Sim2Real ###
        sim2real = AgentArenaInterface(
            workspace_sampler, args.task, config_name, 
            checkpoint, max_steps, 
            adjust_orien=args.adjust_orient, 
            adjust_pick=args.adjust_pick, 
            save_dir=args.save_dir,
            depth_sim2real=args.depth_sim2real,
            mask_sim2real=args.mask_sim2real,
            sim_camera_height=args.sim_camera_height,
            goal_conditioned=args.goal_conditioned,
            whole_workspace=True,
            callback_on_internal_states=visualise_action_inference)
        
        sim2real.run()
    except Exception as e:
        print(f'Caught exception: {e}')
        print('Stack trace:')
        traceback.print_exc()
        if sim2real is not None:
            sim2real.stop_video()
            rclpy.try_shutdown()
            sim2real.destroy_node()