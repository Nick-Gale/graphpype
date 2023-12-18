#!/bin/bash

python3 -m venv venv
source ./venv/bin/activate

wget https://www.github.com/Nick-Gale/NetworkAnalysis/releases/v001.tar.gz

python3 -m install -e v001.tar.gz
