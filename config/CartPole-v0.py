'''
Network configuration for CartPole-v0

Copyright (C) 2020 Richard Herbert and Simon D. Levy

MIT License
'''

# This code gets run by exec() in es-evolve.py, so there are no imports here

net = ArgMaxNet(
    nn.Linear(4, 100),
    nn.ReLU(True),
    nn.Linear(100, 2),
    nn.Softmax(dim=-1))

target = 200
