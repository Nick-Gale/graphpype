#!/bin/bash
# Create a simple development install for the graphpype neural imaging package
python3 -m venv venv
wget -O install.tar.gz https://github.com/Nick-Gale/NetworkAnalysis/raw/main/dist/v0/graphpype-v0.tar.gz 
tar -xzvf install.tar.gz
rm install.tar.gz
source ./venv/bin/activate
python3 -m pip install -e .
