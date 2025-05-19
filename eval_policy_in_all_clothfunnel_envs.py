import sys
import argparse
import os

import agent_arena as ag_ar
from controllers.planet.rssm import RSSM

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    # parser.add_argument('--arena', default='')
    #parser.add_argument('--agent', default='planet')
    parser.add_argument('--config', default='RGB2RGB-on-all-garment-800k')
    parser.add_argument('--model_dir', default='test_results')
    parser.add_argument('--save_dir', default='test_results/evals')
    parser.add_argument('--test_object', default='longsleeve', type=str)
    parser.add_argument('--verbose', default='silence', type=str)
    parser.add_argument('--disp', default=0, type=int)
    parser.add_argument('--eval_checkpoint', default='-1', type=int)

    return parser.parse_args()

def main():
    args = parse_arguments()

    disp = args.disp == 1  # Convert disp to boolean directly

    agent_name = "planet-clothpick++"
    trained_arena = f"softgym|domain:clothfunnels-realadapt-longsleeve,task:flattening,horizon:30"
    
    config = ag_ar.retrieve_config(
        agent_name, 
        trained_arena, 
        args.config,
        config_dir='configuration')
    # create_message_logger(config.save_dir, args.verbose)
    model_dir = os.path.join(args.model_dir, trained_arena, agent_name, args.config)
    #save_dir = os.path.join(args.save_dir, agent_name, args.config, args.test_object)
   
    eval_arena = f"softgym|domain:clothfunnels-realadapt-{args.test_object},task:flattening,horizon:30"
    arena = ag_ar.build_arena(f"{eval_arena},disp:{disp}")
    agent = RSSM(config)

    # arena.set_log_dir(save_dir)
    # agent.set_log_dir(save_dir)

     #agent.set_log_dir(save_dir)
    #agent.load(path=model_dir)
   

    for obj in ['longsleeve', 'pants', 'skirt', 'dress']:
        eval_arena = f"softgym|domain:clothfunnels-realadapt-{obj},task:flattening,horizon:30"
        arena = ag_ar.build_arena(f"{eval_arena},disp:{disp}")
        save_dir = os.path.join(args.save_dir, agent_name, args.config, obj)
        arena.set_log_dir(save_dir)
        agent.set_log_dir(model_dir)
        ag_ar.evaluate(agent, arena, checkpoint=-1)

if __name__ == '__main__':
    main()