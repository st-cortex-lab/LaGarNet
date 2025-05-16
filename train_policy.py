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
    parser.add_argument('--log_dir', default='test_results')
    parser.add_argument('--object', default='longsleeve', type=str)
    parser.add_argument('--verbose', default='silence', type=str)
    parser.add_argument('--disp', default=0, type=int)
    parser.add_argument('--eval_checkpoint', default='-1', type=int)

    return parser.parse_args()

def main():
    args = parse_arguments()

    disp = args.disp == 1  # Convert disp to boolean directly

    agent_name = "planet-clothpick++"
    trained_arena = f"softgym|domain:clothfunnels-realadapt-{args.object},task:flattening,horizon:30"

    config = ag_ar.retrieve_config(
        agent_name, 
        trained_arena, 
        args.config,
        config_dir='configuration')
    # create_message_logger(config.save_dir, args.verbose)
    save_dir = os.path.join(args.log_dir, trained_arena, agent_name, args.config)
    
   

    arena = ag_ar.build_arena(f"{trained_arena},disp:{disp}")
    agent = RSSM(config)

    arena.set_log_dir(save_dir)
    agent.set_log_dir(save_dir)

    validation_interval = config.validation_interval
    total_update_steps = config.total_update_steps


    ag_ar.train_and_evaluate(agent, arena,
        validation_interval, total_update_steps, args.eval_checkpoint)

if __name__ == '__main__':
    main()