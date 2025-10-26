#!/bin/bash
srun --job-name=hopno700 \
     --output=hoppernosurf700_output_%j.log \
     --error=hoppernosurf700_error_%j.log \
     --partition=Abaki \
     --qos=abaki \
     --chdir=/home/d/durmusy/Desktop/GIT/sep-groupb \
    xvfb-run -a python -u main.py --no-synthetic_feedback --total_queries 700 --no-surf
