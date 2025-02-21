sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=RLHF
#SBATCH --comment=RLHF-training
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malikdurmus2@gmail.com
#SBATCH --chdir=/home/d/durmusy/Desktop/GIT/sep-groupb
#SBATCH --output=/home/d/durmusy/Desktop/GIT/sep-groupb/slurm.%j.%N.out
#SBATCH --ntasks=1
#SBATCH --partition=NvidiaAll
#SBATCH --qos=high

module load python/3.8
source venv/bin/activate

# Function to check GPU memory usage every 10 seconds
check_gpu_memory() {
    while true; do
        # Check memory usage
        python -c "
import torch
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0)
    cached = torch.cuda.memory_reserved(0)
    print(f'Memory Allocated: {allocated / 1024**2:.2f} MB')
    print(f'Memory Cached: {cached / 1024**2:.2f} MB')
else:
    print('CUDA not available.')
"
        # Wait 10 seconds before checking again
        sleep 10
    done
}

# Start GPU memory check in the background
check_gpu_memory &

# Run the main training script
xvfb-run -a python -u main.py --synthetic_feedback
EOF
