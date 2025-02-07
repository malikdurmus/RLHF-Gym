cd ..
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=RLFH
#SBATCH --comment=RLFH-training
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malikdurmus2@gmail.com
#SBATCH --chdir=/home/d/durmusy/Desktop/GIT/sep-groupb
#SBATCH --output=/home/d/durmusy/Desktop/GIT/sep-groupb/surf_runs/slurm.%j.%N.out
#SBATCH --ntasks=1
#SBATCH --partition=NvidiaAll

module load python/3.8
source venv/bin/activate
xvfb-run -a python -u main.py --no-surf --no-ensemble_sampling
EOF


