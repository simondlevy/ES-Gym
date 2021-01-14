#!/usr/bin/env python3

'''
Python distutils setup file for es-gym module.

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from setuptools import setup

setup(
    name='es_gym',
    license='MIT',
    description='Evolutionary Strategies using PyTorch',
    url='https://github.com/simondlevy/PyTorch-ES',
    author='Richard Herbert and Simon D. Levy',
    author_email='richard.alan.herbert@gmail.com',
    packages=['es_gym', 'es_gym.strategies', 'es_gym.utils'],
    keywords=['machine learning', 'ai', 'evolutionary strategies',
              'reinforcement learning', 'pytorch']
)
