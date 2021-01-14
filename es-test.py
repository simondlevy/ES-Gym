#!/usr/bin/env python3
'''
ES tester script for OpenAI Gym environemnts

Copyright (C) 2020 Richard Herbert and Simon D. Levy

MIT License
'''

import argparse
import torch
from es_gym import eval_net


def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    parser.add_argument('--record', default=None,
                        help='If specified, sets the recording dir')
    parser.add_argument('--seed', type=int, required=False,
                        help='Seed for random number generator')
    parser.add_argument('--nodisplay', dest='nodisplay', action='store_true',
                        help='Suppress display')
    args = parser.parse_args()

    # Load net and environment name from pickled file
    net, env_name = torch.load(open(args.filename, 'rb'))

    # Evaluate the network
    eval_net(net, env_name,
             render=not args.nodisplay, seed=args.seed, report=True)


if __name__ == '__main__':
    main()
