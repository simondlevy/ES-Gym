'''
PyTorch network for CartPole-v0

Copyright (C) 2020 Richard Herbert and Simon D. Levy

MIT License
'''

net = ArgmaxNet(
    nn.Linear(4, 100),
    nn.ReLU(True),
    nn.Linear(100, 2),
    nn.Softmax()
)
