from setuptools import setup

setup(
    name='pytorch_es',
    license='MIT',
    description='Evolutionary Strategies using PyTorch',
    url='https://github.com/simondlevy/PyTorch-ES',
    author='Richard Herbert and Simon D. Levy',
    author_email='richard.alan.herbert@gmail.com',
    packages=['pytorch_es', 'pytorch_es.strategies', 'pytorch_es.utils'],
    keywords=["machine learning", "ai", "evolutionary strategies", "reinforcement learning", "pytorch"],
)
