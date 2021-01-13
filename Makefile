#
# Makefile for convenience
#
# Copyright (C) 2020 Simon D. Levy
#
# MIT License
#

install:
	sudo python3 setup.py install

evo:
	./es-evolve.py

edit:
	vim es-evolve.py

clean:
	sudo rm -rf build/ dist/ *.egg-info __pycache__ */__pycache__ models/

commit:
	git commit -a

flake:
	flake8 setup.py
	flake8 es-evolve.py
	flake8 es-test.py
	flake8 pytorch_es/*.py
	flake8 pytorch_es/nets/*.py
	flake8 pytorch_es/strategies/*.py
	flake8 pytorch_es/utils/*.py
