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
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import pickle
import time

from gym import logger as gym_logger

import numpy as np

import torch
import torch.nn as nn  # noqa: F401

from es_gym import eval_net
from es_gym.nets import ArgMaxNet, ClipNet  # noqa: F401


class EvolutionModule:
    '''
    Evolutionary Strategies module for PyTorch models -- modified from
    https://github.com/alirezamika/evostra
    '''
    def __init__(
        self,
        weights,
        reward_func,
        population_size=50,
        sigma=0.1,
        learning_rate=0.001,
        decay=1.0,
        sigma_decay=1.0,
        cuda=False,
        reward_goal=None,
        consecutive_goal_stopping=None,
        save_path=None
    ):
        np.random.seed(int(time.time()))
        self.weights = weights
        self.reward_function = reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        self.cuda = cuda
        self.decay = decay
        self.sigma_decay = sigma_decay
        self.pool = ThreadPool(mp.cpu_count())
        self.reward_goal = reward_goal
        self.consecutive_goal_stopping = consecutive_goal_stopping
        self.consecutive_goal_count = 0
        self.save_path = save_path

    def jitter_weights(self, weights, population=[], no_jitter=False):
        new_weights = []
        for i, param in enumerate(weights):
            if no_jitter:
                new_weights.append(param.data)
            else:
                jittered = torch.from_numpy(self.SIGMA *
                                            population[i]).float()
                if self.cuda:
                    jittered = jittered.cuda()
                new_weights.append(param.data + jittered)
        return new_weights

    def run(self, iterations, print_step=1):

        for iteration in range(iterations):

            population = []
            for _ in range(self.POPULATION_SIZE):
                x = []
                for param in self.weights:
                    x.append(np.random.randn(*param.data.size()))
                population.append(x)

            rewards = self.pool.map(
                self.reward_function,
                [self.jitter_weights(copy.deepcopy(self.weights),
                                     population=pop) for pop in population]
            )
            if np.std(rewards) != 0:
                normalized_rewards = ((rewards - np.mean(rewards)) /
                                      np.std(rewards))
                for index, param in enumerate(self.weights):
                    A = np.array([p[index] for p in population])
                    rewards_pop = (
                            torch.from_numpy(np.dot(A.T,
                                             normalized_rewards).T).float())
                    if self.cuda:
                        rewards_pop = rewards_pop.cuda()
                    param.data = (param.data + self.LEARNING_RATE /
                                  (self.POPULATION_SIZE * self.SIGMA) *
                                  rewards_pop)

                    self.LEARNING_RATE *= self.decay
                    self.SIGMA *= self.sigma_decay

            if (iteration+1) % print_step == 0:
                test_reward = self.reward_function(
                    self.jitter_weights(copy.deepcopy(self.weights),
                                        no_jitter=True))
                print('iter %6d. reward: %+.3f' % (iteration+1, test_reward))

                if self.save_path:
                    pickle.dump(self.weights, open(self.save_path, 'wb'))

                if self.reward_goal and self.consecutive_goal_stopping:
                    if test_reward >= self.reward_goal:
                        self.consecutive_goal_count += 1
                    else:
                        self.consecutive_goal_count = 0

                    if (self.consecutive_goal_count >=
                            self.consecutive_goal_stopping):
                        return self.weights

        return self.weights


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
    parser.add_argument('--seed', type=int, required=False,
                        help='Seed for random number generator')
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--target', type=float, default=None,
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

    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)

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

        return eval_net(cloned_net, args.env, seed=args.seed)

    partial_func = partial(get_reward, net=net)
    mother_parameters = list(net.parameters())

    es = EvolutionModule(
        mother_parameters,
        partial_func,
        population_size=args.pop,
        sigma=args.sigma,
        learning_rate=args.lr,
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
