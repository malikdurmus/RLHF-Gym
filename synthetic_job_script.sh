#!/bin/bash
#SBATCH --job-name=RLFH
#SBATCH --comment=RLFH-training
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email_adress@mail.com
#SBATCH --chdir=your_project_directory
#SBATCH --output=your_project_directory/slurm.%j.%N.out
#SBATCH --ntasks=1
#SBATCH --partition=NvidiaAll

# Load necessary modules 3.8, 3.9 or 3.10
module load python/3.8

# Activate virtual environment
source venv/bin/activate

# Adjust args use only for synthetic feedback
python main.py --synthetic-feedback
