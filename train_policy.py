import sys
import argparse
import os
import numpy as np
import torch

import agent_arena as ag_ar
from controllers.planet.rssm import RSSM
from controllers.gc_rssm import GC_RSSM


def reward_bonus_and_penalty(rewards, observations, actions):
    if isinstance(rewards, torch.Tensor):
        rewards_ = rewards.clone()
    else:
        rewards_ = rewards.copy()

    above_0_9 = observations['normalised_coverage'][:, :-1] > 0.9
    below_0_9 = observations['normalised_coverage'][:, 1:] < 0.9
    first_state_no_term = observations['terminal'][:, :-1] == 0

    rewards_[:, 1:][above_0_9 & below_0_9 & first_state_no_term] = 0

    above_0_9_5 = observations['normalised_coverage'][:] > 0.95
    rewards_[above_0_9_5] = 0.7


    return rewards

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    # parser.add_argument('--arena', default='')
    #parser.add_argument('--agent', default='planet')
    parser.add_argument('--config', default='RGB2RGB-on-all-garment-800k')
    parser.add_argument('--wm', default='rssm', type=str)
    parser.add_argument('--log_dir', default='test_results/models')
    parser.add_argument('--object', default='longsleeve', type=str)
    parser.add_argument('--verbose', default='silence', type=str)
    parser.add_argument('--disp', default=0, type=int)
    parser.add_argument('--eval_checkpoint', default='-1', type=int)
    ## following is boolean

    return parser.parse_args()

def main():
    args = parse_arguments()

    disp = args.disp == 1  # Convert disp to boolean directly

    #agent_name = "planet-clothpick++"
    trained_arena = f"softgym|domain:clothfunnels-realadapt-{args.object},task:flattening,horizon:30"

    config = ag_ar.retrieve_config(
        args.wm, 
        trained_arena, 
        args.config,
        config_dir='configuration')
    # create_message_logger(config.save_dir, args.verbose)
    save_dir = os.path.join(args.log_dir, args.wm, trained_arena, args.config)
    
   

    arena = ag_ar.build_arena(f"{trained_arena},disp:{disp}")
    if args.wm == 'rssm':
        agent = RSSM(config)
    elif args.wm == 'gc_rssm':
        agent = GC_RSSM(config)
    else:
        raise ValueError(f"Unknown world model: {args.wm}")
    
    if config.get('apply_transform_in_dataset', False):
        agent.set_reward_processor(reward_bonus_and_penalty)

    arena.set_log_dir(save_dir)
    agent.set_log_dir(save_dir)

    validation_interval = config.validation_interval
    total_update_steps = config.total_update_steps


    ag_ar.train_and_evaluate(agent, arena,
        validation_interval, total_update_steps, args.eval_checkpoint)

if __name__ == '__main__':
    main()