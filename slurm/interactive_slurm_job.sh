#!/bin/bash
cd ..
srun --job-name=python_interactive_job \
     --output=output_%j.log \
     --error=error_%j.log \
     --partition=NvidiaAll \
    xvfb-run -a python -u main.py --no-synthetic_feedback
