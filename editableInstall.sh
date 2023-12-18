#!/bin/bash

python3 -m venv venv
source ./venv/bin/activate

wget https://www.github.com/Nick-Gale/NetworkAnalysis/dist/v0/graphpype-v0.0.1.tar.gz

python3 -m install -e v001.tar.gz
