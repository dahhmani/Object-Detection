#!/usr/bin/env sh

. ../venv/bin/activate
pip freeze --local | grep -v "pkg-resources" > ../dependencies.txt
