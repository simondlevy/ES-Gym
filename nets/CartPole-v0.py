'''
PyTorch network for CartPole-v0

Copyright (C) 2020 Richard Herbert and Simon D. Levy

MIT License
'''

#import torch
#from torch.autograd import Variable
#import torch.nn as nn

net = nn.Sequential(
    nn.Linear(4, 100),
    nn.ReLU(True),
    nn.Linear(100, 2),
    nn.Softmax()
)
