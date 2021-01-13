#!/usr/bin/env python3
'''
Network classes for supporting activation functions

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np
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
