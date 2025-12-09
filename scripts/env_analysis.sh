#!/bin/bash

sudo apt install python3.10-venv

python3 -m venv env_analysis

. env_analysis/bin/activate && pip install -r analysis/requirements.txt