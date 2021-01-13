#!/usr/bin/env python3
'''
ES evolver script for OpenAI Gym environemnts

Copyright (C) 2020 Richard Herbert and Simon D. Levy

MIT License
'''

import argparse
import copy
from functools import partial
import logging
import os

from gym import logger as gym_logger

import numpy as np

import torch
import torch.nn as nn  # noqa: F401

from pytorch_es import EvolutionModule
from pytorch_es.utils.helpers import run_net
from pytorch_es.nets import ArgMaxNet, ClipNet  # noqa: F401


def main():

    gym_logger.setLevel(logging.CRITICAL)

    fmtr = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmtr)
    parser.add_argument('--env', default='CartPole-v0',
                        help='Environment id')
    parser.add_argument('--cuda', action='store_true',
                        help='Whether or not to use CUDA')
    parser.add_argument('--pop', type=int, default=64, help='Population size')
    parser.add_argument('--iter', type=int, default=400, help='Iterations')
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--threads', type=int, default=15,
                        help='Thread count')
    parser.add_argument('--target', type=float, default=-np.inf,
                        help='Reward target')
    parser.add_argument('--csg', type=int, default=10,
                        help='Consecutive goal stopping')
    args = parser.parse_args()

    cuda = False

    if args.cuda:
        if torch.cuda.is_available():
            cuda = True
        else:
            print('******* Sorry, CUDA not available *******')

    # Run code in script named by environment
    code = open('./nets/%s.py' % args.env).read()
    ldict = {}
    exec(code, globals(), ldict)
    net = ldict['net']

    if cuda:
        net = net.cuda()

    def copy_weights_to_net(weights, net):

        for i, param in enumerate(net.parameters()):
            try:
                param.data.copy_(weights[i])
            except Exception:
                param.data.copy_(weights[i].data)

    def get_reward(weights, net):

        cloned_net = copy.deepcopy(net)

        copy_weights_to_net(weights, cloned_net)

        return run_net(cloned_net, args.env)

    partial_func = partial(get_reward, net=net)
    mother_parameters = list(net.parameters())

    es = EvolutionModule(
        mother_parameters,
        partial_func,
        population_size=args.pop,
        sigma=args.sigma,
        learning_rate=args.lr,
        threadcount=args.threads,
        cuda=cuda,
        reward_goal=args.target,
        consecutive_goal_stopping=args.csg)

    final_weights = es.run(args.iter)

    # Save final weights in a new network, along with environment name
    os.makedirs('models', exist_ok=True)
    reward = partial_func(final_weights)
    copy_weights_to_net(final_weights, net)
    filename = 'models/%s%+.3f.dat' % (args.env, reward)
    print('Saving %s' % filename)
    torch.save((net, args.env), open(filename, 'wb'))


if __name__ == '__main__':
    main()
