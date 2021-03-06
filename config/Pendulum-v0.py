'''
Network configuration for Pendulum-v0

Copyright (C) 2020 Richard Herbert and Simon D. Levy

MIT License
'''

# This code gets run by exec() in es-evolve.py, so there are no imports here
net = ClipNet(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh())
