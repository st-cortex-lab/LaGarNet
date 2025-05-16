import argparse

import agent_arena.api as ag_ar
from planet_clothpick_plus_plus.controllers.workspace_sampler_planet_clothpick \
    import WorkspaceSamplerPlaNetClothPick
from dotmap import DotMap
import os
from agent_arena.utilities.perform_single import perform_single
from agent_arena.utilities.visual_utils import \
    draw_pick_and_place, plot_image_trajectory
import numpy as np
from matplotlib import cm

# python evaluate.py --log_dir /data/fast-ldm-fabric-shaper --agent diffusion_policy 
# --config masked-depth-human-50 --log_dir /data/fast-ldm-fabric-shaper --eval_checkpoint 99999

def main():

    ### Argument Parsing
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent_config',  default='GC-RGB2M-on-mix-100k-sf-goal-rotate-translate')
    parser.add_argument('--log_dir', default='/data/planet-clothpick++')
    parser.add_argument('--verbose', default='silence', type=str)
    parser.add_argument('--disp', default=0, type=int)
    parser.add_argument('--eval_checkpoint', default='-1', type=int)
    parser.add_argument('--object', default='longsleeve', type=str)
    parser.add_argument('--sampler_config', default='cloth-mask-centre-multiple-prioritise-middle', type=str)
    
    args = parser.parse_args()

    if args.disp == 1:
        disp = True
    elif args.disp == 0:
        disp = False
    
    agent_name = "planet-clothpick"
    trained_arena = f"softgym|domain:clothfunnels-realadapt-{args.object},task:flattening,horizon:30"
    # capitalise the first letter of object
    capitalised_obj = args.object[0].upper() + args.object[1:]
    test_arena = f"softgym|domain:ur3e-small{capitalised_obj},task:flattening,horizon:8"
    

    config = ag_ar.retrieve_config(
        agent_name, 
        trained_arena, 
        args.agent_config)
    #print(config)
    
    # This is for UR3e environment
    config.transform.params.depth_clip_min = 0.55 #0.72
    config.transform.params.depth_clip_max = 0.75 #0.76
    config.transform.params.depth_min = 0.55 # 0.72
    config.transform.params.depth_max = 0.75 # 0.76
   
    #create_message_logger(config.save_dir, args.verbose)
    #print(config)
    arena = ag_ar.build_arena(test_arena + ',disp:{}'.format(disp))
    agent = ag_ar.build_agent(agent_name, config)
    agent.set_eval()
    #logger = ag_ar.build_logger(arena.logger_name, config.save_dir)
    save_dir = os.path.join(
        args.log_dir, test_arena, f'ws_smplr_{agent_name}', f'{args.agent_config}+{args.sampler_config}')
    model_dir = os.path.join(args.log_dir, trained_arena, f'{agent_name}', f'{args.agent_config}')
    arena.set_log_dir(save_dir)
    agent.set_log_dir(model_dir)
    

    config_file = os.path.join('configs', args.sampler_config + '.yaml')
    import yaml
    with open(config_file, 'r') as f:
        worspace_sampler_config = yaml.safe_load(f)

    worspace_sampler_config['agent'] = agent
    worspace_sampler_config['checkpoint'] = int(args.eval_checkpoint)
    worspace_sampler_config['policy'] = {'params': config.policy.params}

    # worspace_sampler_config = {
    #     'agent': agent,
    #     'checkpoint': args.eval_checkpoint,
    #     'offset_left_ratio': 0,
    #     'offset_right_ratio': 0,
    #     'sample_interval_ratio': 0,
    #     'policy': {'params': config.policy.params}
    # }

    worspace_sampler_config = DotMap(worspace_sampler_config)
    

    workspace_sampler = WorkspaceSamplerPlaNetClothPick(worspace_sampler_config)
    workspace_sampler.set_log_dir(save_dir)

    eid = 5
    episode_eval_configs = arena.get_eval_configs()
    res = perform_single(arena, workspace_sampler, mode='eval', 
            episode_config=episode_eval_configs[eid], collect_frames=False, end_success=False)

    internal_states = res['internal_states']

    print('internal_states', internal_states[0].keys())

    interation_means = [its['iteration_means'] for its in internal_states] # T * I * 1 * 4
    interation_means = np.stack(interation_means, axis=0)
    print('iteration_means', interation_means.shape)
    input_obs = [its['raw_input_obs'] for its in internal_states]
    input_obs = np.stack(input_obs, axis=0)[:, :, :, :3]
    print('input_obs', input_obs.shape)

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
    
    plot_image_trajectory(draw_action_img, save_path='tmp/visualise_inference', title='iteration')


    

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
        img = input_obs[i]
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
    
    plot_image_trajectory(draw_action_img, save_path='tmp/visualise_inference', title='last_step_reward')

    pick_masks = [its['pick-mask'] for its in internal_states]
    place_masks = [its['place-mask'] for its in internal_states]

    plot_image_trajectory(pick_masks, save_path='tmp/visualise_inference', title='pick_masks')
    plot_image_trajectory(place_masks, save_path='tmp/visualise_inference', title='place_masks')




    


    
    
if __name__ == '__main__':
    main()