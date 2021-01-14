"""Helpers for PyTorch-ES examples"""

import time
import numpy as np
import gym
import torch
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


def eval_net(net, env_name, render=False, seed=None):

    # Make environment from name
    env = gym.make(env_name)

    env.seed(seed)

    # Run net on environment
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
            time.sleep(0.02)
        batch = torch.from_numpy(ob[np.newaxis, ...]).float()
        prediction = net(Variable(batch))
        action = net.actfun(prediction.data.numpy()[0])
        ob, reward, done, _ = env.step(action)

        total_reward += reward
    env.close()

    return total_reward
