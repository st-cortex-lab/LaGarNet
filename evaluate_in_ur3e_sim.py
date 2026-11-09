import argparse

import actoris_harena.api as ag_ar
from planet_clothpick_plus_plus.controllers.workspace_sampler_planet_clothpick \
    import WorkspaceSamplerPlaNetClothPick
from dotmap import DotMap
import os

# python evaluate.py --log_dir /data/fast-ldm-fabric-shaper --agent diffusion_policy 
# --config masked-depth-human-50 --log_dir /data/fast-ldm-fabric-shaper --eval_checkpoint 99999

def main():

    ### Argument Parsing
    parser = argparse.ArgumentParser()
    # parser.add_argument('--trained_arena', 
    #     default="softgym|domain:clothfunnels-real2sim-longsleeve,task:flattening,horizon:30")
    # parser.add_argument('--test_arena',
    #     default="softgym|domain:ur3e-smallLongsleeve,task:flattening,horizon:30")
    # parser.add_argument('--agent', 
    #     default="planet-clothpick")
    parser.add_argument('--agent_config',  default='MD2M-on-mix-200k-sf')
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
    test_arena = f"softgym|domain:ur3e-small{capitalised_obj},task:flattening,horizon:30"
    

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
    
    # worspace_sampler_config = {
    #     'agent': agent,
    #     'checkpoint': args.eval_checkpoint,
    #     'offset_left_ratio': 0.1,
    #     'offset_right_ratio': 0.1,
    #     'sample_interval_ratio': 0.02,
    #     'policy': {'params': config.policy.params}
    # }
    
    # read from config file in configs/args.sampler_config.yaml

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

    ag_ar.evaluate(workspace_sampler, arena, args.eval_checkpoint)
    
if __name__ == '__main__':
    main()