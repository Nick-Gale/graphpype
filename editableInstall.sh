#!/bin/bash
# Create a simple development install for the graphpype neural imaging package
python3 -m venv venv
source ./venv/bin/activate
wget https://www.github.com/Nick-Gale/NetworkAnalysis/dist/v0/graphpype-v0.0.1.tar.gz
tar -xzvf graphpype-v0.0.1.tar.gz
python3 -m install -e .
