import time

import numpy as np

import gym

import torch
from torch.autograd import Variable
import torch.nn as nn


class _NetWithActfun(nn.Sequential):

    def __init__(self, *args):

        nn.Sequential.__init__(self, *args)


class ArgMaxNet(_NetWithActfun):

    def __init__(self, *args):

        _NetWithActfun.__init__(self, *args)

    def actfun(self, x):

        return np.argmax(x)


class ClipNet(_NetWithActfun):

    def __init__(self, *args):

        _NetWithActfun.__init__(self, *args)

    def actfun(self, x):

        return np.clip(x, -1, +1)


def eval_net(net, env_name, render=False, seed=None, report=False):

    # Make environment from name
    env = gym.make(env_name)

    env.seed(seed)

    # Run net on environment
    ob = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        if render:
            env.render()
            time.sleep(0.02)
        batch = torch.from_numpy(ob[np.newaxis, ...]).float()
        prediction = net(Variable(batch))
        action = net.actfun(prediction.data.numpy()[0])
        ob, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1

    env.close()

    if report:
        print('Got reward %+6.6f in %d steps' % (total_reward, steps))

    return total_reward
