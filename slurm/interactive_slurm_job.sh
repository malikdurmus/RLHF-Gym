#!/bin/bash
PROJECT_DIR = ~/sep-groupb
cd $PROJECT_DIR

srun --job-name=python_interactive_job \
     --output=output_%j.log \
     --error=error_%j.log \
     --partition=NvidiaAll \
     --chdir=/home/d/durmusy/Desktop/GIT/sep-groupb \
    xvfb-run -a python -u main.py --no-synthetic_feedback
