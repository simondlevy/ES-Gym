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
import time

from pytorch_es import EvolutionModule
import gym
from gym import logger as gym_logger
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

def main():

    gym_logger.setLevel(logging.CRITICAL)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='CartPole-v0', help='Environment id')
    parser.add_argument('--cuda', action='store_true', help='Whether or not to use CUDA')
    parser.add_argument('--pop', type=int, default=5, help='Population size')
    parser.add_argument('--iter', type=int, default=400, help='Iterations')
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--threads', type=int, default=15, help='Thread count')
    parser.add_argument('--target', type=float, default=-np.inf, help='Reward target')
    parser.add_argument('--csg', type=int, default=10, help='Consecutive goal stopping')
    args = parser.parse_args()

    cuda = False

    if args.cuda:
        if torch.cuda.is_available():
            cuda = True
        else:
            print('******* Sorry, CUDA not available *******')

    net = nn.Sequential(
        nn.Linear(4, 100),
        nn.ReLU(True),
        nn.Linear(100, 2),
        nn.Softmax()
    )

    if cuda:
        net = net.cuda()

    def get_reward(weights, net, render=False):
        cloned_net = copy.deepcopy(net)
        for i, param in enumerate(cloned_net.parameters()):
            try:
                param.data.copy_(weights[i])
            except:
                param.data.copy_(weights[i].data)

        env = gym.make(args.env)
        ob = env.reset()
        done = False
        total_reward = 0
        while not done:
            if render:
                env.render()
                time.sleep(0.05)
            batch = torch.from_numpy(ob[np.newaxis,...]).float()
            if cuda:
                batch = batch.cuda()
            prediction = cloned_net(Variable(batch))
            action = prediction.data.numpy().argmax()
            ob, reward, done, _ = env.step(action)

            total_reward += reward 

        env.close()
        return total_reward
        
    partial_func = partial(get_reward, net=net)
    mother_parameters = list(net.parameters())

    es = EvolutionModule(
        mother_parameters, partial_func, population_size=args.pop, sigma=args.sigma, 
        learning_rate=args.lr, threadcount=args.threads, cuda=cuda, reward_goal=args.target,
        consecutive_goal_stopping=args.csg
    )
    final_weights = es.run(args.iter)

    # Make directory for saving
    os.makedirs('solutions', exist_ok=True)

    reward = partial_func(final_weights)
    filename = 'solutions/%s%+.3f.dat' % (args.env, reward)
    print('Saving %s' % filename)
    torch.save(final_weights, open(filename, 'wb'))

    #reward = partial_func(final_weights, render=True)
    #print(f'Reward from final weights: {reward}')

if __name__ == '__main__':
    main()
