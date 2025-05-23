import sys
import argparse
import os
import numpy as np
import torch

import agent_arena as ag_ar
from controllers.planet.rssm import RSSM


def reward_bonus_and_penalty(rewards, observations, actions):
    # print('reward shape', rewards.shape)
    # print('observation keys()', observations.keys())
    # print('action shape', actions.shape)
    # print('normalised_coverage shape', observations['normalised_coverage'].shape)
    # print('coverage_differance shape', observations['coverage_differance'].shape)

    ## data is in shape B, T, ...

    # print('rewards', rewards[0][:5])
    # print('normalised_coverage', observations['normalised_coverage'][0][:5])
    # print('coverage_differance', observations['coverage_differance'][0][:5])

    ## for steps where normalised_coverage_t > 0.9 but normalised_coverg_{t+1} < 0.9
    # make the reward = 0

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
    # print('reward max', rewards.max())
    # print('reward min', rewards.min())
    # print('reward mean', rewards.mean())
    # print('reward std', rewards.std())
    # if isinstance(rewards, torch.Tensor):
    #     print('reward median', rewards.median())
    # print('before rewards', rewards[0][1:6].flatten())
    # print('after rewards', rewards_[0][1:6].flatten())
    # print('normalised_coverage', observations['normalised_coverage'][0][:6].flatten())

          




    return rewards

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
    ## following is boolean

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