#!/usr/bin/env python3
'''
ES tester script for OpenAI Gym environemnts

Copyright (C) 2020 Richard Herbert and Simon D. Levy

MIT License
'''

import argparse
import time
import torch
from torch.autograd import Variable
import gym
import numpy as np
from pytorch_es.nets import ArgmaxNet

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    parser.add_argument('--record', default=None, help='If specified, sets the recording dir')
    args = parser.parse_args()

    # Load net and environment name from pickled file
    net, env_name = torch.load(open(args.filename, 'rb'))

    # Make environment from name
    env = gym.make(env_name)

    # Run net on environment
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render()
        time.sleep(0.02)
        batch = torch.from_numpy(ob[np.newaxis,...]).float()
        prediction = net(Variable(batch))
        action = net.actfun(prediction.data.numpy())
        ob, reward, done, _ = env.step(action)

        total_reward += reward 
    env.close()

    print(total_reward)

if __name__ == '__main__':
    main()
