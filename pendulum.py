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
import time

import gym
from gym import logger as gym_logger

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def eval_net(net, env_name):

    # Make environment from name
    env = gym.make(env_name)

    # Run net on environment
    ob = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        batch = torch.from_numpy(ob[np.newaxis, ...]).float()
        prediction = net(Variable(batch))
        action = net(prediction.data.numpy()[0])
        ob, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1

    env.close()

    return total_reward, steps, net

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, nhid):

        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, nhid)
        self.l2 = nn.Linear(nhid, nhid)
        self.l3 = nn.Linear(nhid, action_dim)
        self.max_action = max_action

    def forward(self, state):

        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))



def _save_net(net, env_name, reward):
    filename = 'models/%s%+.3f.dat' % (env_name, reward)
    print('Saving %s' % filename)
    torch.save((net, env_name), open(filename, 'wb'))


def _copy_weights_to_net(weights, net):

    for i, param in enumerate(net.parameters()):
        try:
            param.data.copy_(weights[i])
        except Exception:
            param.data.copy_(weights[i].data)


class EvolutionModule:
    '''
    Evolutionary Strategies module for PyTorch models -- modified from
    https://github.com/alirezamika/evostra
    '''
    def __init__(
        self,
        weights,
        reward_func,
        save_name=None,
        population_size=50,
        sigma=0.1,
        learning_rate=0.001,
        decay=1.0,
        sigma_decay=1.0,
        reward_goal=None,
        consecutive_goal_stopping=None,
    ):
        np.random.seed(int(time.time()))
        self.weights = weights
        self.reward_function = reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        self.decay = decay
        self.sigma_decay = sigma_decay
        self.pool = ThreadPool(mp.cpu_count())
        self.reward_goal = reward_goal
        self.consecutive_goal_stopping = consecutive_goal_stopping
        self.consecutive_goal_count = 0
        self.save_name = save_name

    def jitter_weights(self, weights, population=[], no_jitter=False):
        new_weights = []
        for i, param in enumerate(weights):
            if no_jitter:
                new_weights.append(param.data)
            else:
                jittered = torch.from_numpy(self.SIGMA *
                                            population[i]).float()
                new_weights.append(param.data + jittered)
        return new_weights

    def run(self, iterations, target, report_step=1):

        total_steps = 0
        best_reward = -np.inf

        for iteration in range(iterations):

            population = []
            for _ in range(self.POPULATION_SIZE):
                x = []
                for param in self.weights:
                    x.append(np.random.randn(*param.data.size()))
                population.append(x)

            print(self.reward_function(self.weights))
            exit(0)

            results = self.pool.map(
                self.reward_function,
                [self.jitter_weights(copy.deepcopy(self.weights),
                                     population=pop) for pop in population]
            )

            rewards = [r[0] for r in results]

            if np.std(rewards) != 0:
                normalized_rewards = ((rewards - np.mean(rewards)) /
                                      np.std(rewards))
                for index, param in enumerate(self.weights):
                    A = np.array([p[index] for p in population])
                    rewards_pop = (
                            torch.from_numpy(np.dot(A.T,
                                             normalized_rewards).T).float())
                    param.data = (param.data + self.LEARNING_RATE /
                                  (self.POPULATION_SIZE * self.SIGMA) *
                                  rewards_pop)

                    self.LEARNING_RATE *= self.decay
                    self.SIGMA *= self.sigma_decay

            if (iteration+1) % report_step == 0:

                test_reward, steps, net = self.reward_function(
                    self.jitter_weights(copy.deepcopy(self.weights),
                                        no_jitter=True))

                total_steps += steps

                print('Iteration %07d:\treward = %+6.3f%s  \tevaluations = %d'
                      %
                      (iteration+1,
                       test_reward,
                       ' *'
                       if (target is not None) and (test_reward >= target)
                       else '',
                       steps))

                if test_reward > best_reward:
                    best_reward = test_reward
                    if self.save_name is not None:
                        _save_net(net, self.save_name, test_reward)

                if self.reward_goal and self.consecutive_goal_stopping:
                    if test_reward >= self.reward_goal:
                        self.consecutive_goal_count += 1
                    else:
                        self.consecutive_goal_count = 0

                    if (self.consecutive_goal_count >=
                            self.consecutive_goal_stopping):
                        return self.weights

        return self.weights, total_steps


def main():

    gym_logger.setLevel(logging.CRITICAL)

    fmtr = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmtr)
    parser.add_argument('--env', default='Pendulum-v0',
                        help='Environment id')
    parser.add_argument('--pop', type=int, default=64, help='Population size')
    parser.add_argument('--iter', type=int, default=400, help='Iterations')
    parser.add_argument('--seed', type=int, required=False,
                        help='Seed for random number generator')
    parser.add_argument('--checkpoint', action='store_true',
                        help='Save at each new best')
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--target', type=float, default=None,
                        help='Reward target')
    parser.add_argument('--csg', type=int, default=10,
                        help='Consecutive goal stopping')
    args = parser.parse_args()

    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        
    net = Actor(3, 1, 2, 64)

    target = -120

    def get_reward(weights, net):
        cloned_net = copy.deepcopy(net)
        _copy_weights_to_net(weights, cloned_net)
        return eval_net(cloned_net, args.env)

    partial_func = partial(get_reward, net=net)
    mother_parameters = list(net.parameters())

    es = EvolutionModule(
        mother_parameters,
        partial_func,
        population_size=args.pop,
        sigma=args.sigma,
        learning_rate=args.lr,
        reward_goal=target,
        consecutive_goal_stopping=args.csg,
        save_name=(args.env if args.checkpoint else None))

    os.makedirs('models', exist_ok=True)

    final_weights, total_steps = es.run(args.iter, target)

    print('Total evaluations = %d' % total_steps)

    # Save final weights in a new network, along with environment name
    reward = partial_func(final_weights)[0]
    _copy_weights_to_net(final_weights, net)
    _save_net(net, args.env, reward)


if __name__ == '__main__':
    main()
