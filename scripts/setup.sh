#!/usr/bin/env sh

# echo 'alias python=python3' >> ~/.bashrc
# echo 'alias pip=pip3' >> ~/.bashrc
# . ~/.bashrc

python -m venv ../venv/
. ../venv/bin/activate
pip install --upgrade --no-cache-dir pip setuptools
pip install -r ../dependencies.txt

mkdir -p ../data/output/frames ../data/output/video
