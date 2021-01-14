#
# Makefile for convenience
#
# Copyright (C) 2020 Simon D. Levy
#
# MIT License
#

install:
	sudo python3 setup.py install

test:
	rm -rf models/
	./es-evolve.py --target 200
	./es-test.py models/*

edit:
	vim es-evolve.py

help:
	./es-evolve.py --help

clean:
	sudo rm -rf build/ dist/ *.egg-info __pycache__ */__pycache__ */*/__pycache__ models/

commit:
	git commit -a

flake:
	flake8 setup.py
	flake8 es-evolve.py
	flake8 es-test.py
	flake8 es_gym/*.py
	flake8 es_gym/nets/*.py
