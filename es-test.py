#!/usr/bin/env python3
'''
ES tester script for OpenAI Gym environemnts

Copyright (C) 2020 Richard Herbert and Simon D. Levy

MIT License
'''

import argparse
import torch
from pytorch_es.utils.helpers import run_net

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    parser.add_argument('--record', default=None, help='If specified, sets the recording dir')
    args = parser.parse_args()

    # Load net and environment name from pickled file
    net, env_name = torch.load(open(args.filename, 'rb'))

    print('Total reward = %+.3f ' % run_net(net, env_name, render=True))

if __name__ == '__main__':
    main()
