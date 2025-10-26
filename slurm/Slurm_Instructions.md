# **How to Connect to SSH, Set Up a Virtual Environment, and Install Requirements**

## **Connect to SSH **

To connect to the remote server, use the following SSH command: 

```bash 

ssh <cip-kennung>@remote.cip.ifi.lmu.de

```
If you can not connect, please refer to the internal documentation of the RGB:
[SSH](https://www.rz.ifi.lmu.de/infos/ssh_de.html)

You can also use the PyCharm Remote Development tool for this



Use git clone to clone the project, then navigate to your project home directory.

## 2. Create a Virtual Environment with Python 3.8
``` bash 
python3.8 -m venv myenv
```
## 3. Activate the Virtual Environment
```
source myenv/bin/activate
```
## 4. Install the Required Packages
``` bash
pip install -r slurm/requirements.txt
```

Now you can run jobs on SLURM
# **Running Jobs on Slurm**

  
This guide provides instructions for submitting both interactive and non-interactive jobs on a Slurm-managed cluster. It covers how to request resources, check job statuses, and troubleshoot.

## **1. Interactive Jobs (Human Feedback) with `srun`**

### **Purpose**

Use `srun` when you want to start an interactive session on a compute node. This is useful for tasks such as debugging or running scripts with immediate output.
+ *You can also use the computer that you connect with ssh to run the program interactively, but using this gives you the advantage to parallelize the interactive sessions*
+ E.g: run the script in multiple compute clusters with different parameters, give feedback to computer1, while waiting for computer1 to ask for feedback again, switch to computer2 and give feedback. while waiting for computer2 to ask for feedback again, switch to computer1 (... this goes on until the computers finish)


### **Command for Interactive Jobs**

```bash
#!/bin/bash

srun --job-name=python_interactive_job \
     --output=output_%j.log \
     --error=error_%j.log \
     --partition=NvidiaAll \
    xvfb-run -a python -u main.py

```

- `--job-name=<job_name>`: Name of the job.

- `--partition=NvidiaAll`: Partition to request.

- `xvfb-run -a python -u main.py`: Use xvfb (virtual screen) and run python main

#### Step by step instructions

After running srun for human feedback, SLURM will allocate resources for you:

```bash
(venv) durmusy@datolith:Desktop/GIT/sep-groupb (130) [18:33:48] % srun --job-name=human_feedback_job \    
     --output=output_%j.log \
     --error=error_%j.log \
     --partition=NvidiaAll \
    xvfb-run -a python -u main.py
srun: job 709536 queued and waiting for resources
srun: job 709536 has been allocated resources
```

You can examine the output.<your_job_id> file to examine your program.

After your job has been allocated resources, open a new terminal and ping the allocated computer to get the IP of the remote compute cluster:
```bash
(venv) durmusy@datolith:Desktop/GIT/sep-groupb (0) [18:33:10] % squeue -u $USER                                          
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            709536 NvidiaAll human_fe  durmusy  R       3:39      1 aleurit
%                                                                                                                                                                                                                                   (venv) durmusy@datolith:Desktop/GIT/sep-groupb (0) [18:44:30] % ping aleurit 
PING aleurit(aleurit.cip.ifi.lmu.de (2001:4ca0:4000:0:141:84:220:65)) 56 data bytes
64 bytes from aleurit.cip.ifi.lmu.de (2001:4ca0:4000:0:141:84:220:65): icmp_seq=1 ttl=64 time=0.219 ms
^C

```

Now you have the ip adress, in this case it is: aleurit.cip.ifi.lmu.de (2001:4ca0:4000:0:141:84:220:65)  

Forward the port by executing: 
```bash
ssh -L 6001:127.0.0.1:5000 durmusy@aleurit.cip.ifi.lmu.de
```

Now you can go to the adress: (http://127.0.0.1:6001/) to access the application

**Notes**
+ You can run the application in multiple SLURM computers by executing srun command above multiple times in different terminals
+ You need to forward to a different port for each app:
+ Lets say you forward the port 5000 of the computer aleurit to 6001
+ and port 5000 of the computer dolomit to 6002 
+ and port 5000 of the computer andesit to 6003

Now you have three seperate apps on 
+ (http://127.0.0.1:6001/) 
+ (http://127.0.0.1:6002/) 
+ (http://127.0.0.1:6003/) 

You can go to those adresses and give feedback. All 3 apps are individual and do not interfere with each other. you can run them with different hyperparameters to understand how they influence our program.
## **2. Non-Interactive Jobs (Synthetic) with `sbatch`**

### **Purpose**

Use `sbatch` when you want to submit a job to Slurm in the background, especially for synthetic feedback that doesn’t require interaction.
### **Basic Command for Non-Interactive Jobs**

Use the batch script provided in the projects' slurm directory (`synthetic_job_script.sh`) with the following content:

```bash
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=RLHF
#SBATCH --comment=RLHF-training
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malikdurmus2@gmail.com
#SBATCH --chdir=/home/d/durmusy/Desktop/GIT/sep-groupb
#SBATCH --output=/home/d/durmusy/Desktop/GIT/sep-groupb/surf_runs/slurm.%j.%N.out
#SBATCH --ntasks=1
#SBATCH --partition=NvidiaAll

module load python/3.8
source venv/bin/activate
xvfb-run -a python -u main.py
EOF

```

- `--output=output_%j.log`: Log file for standard output (`%j` is replaced with the job ID).

- `--error=error_%j.log`: Log file for error output.
 

- **replace the following**: 
+ --mail-user= malikdurmus2@gmail.com
+ --chdir= /home/d/durmusy/Desktop/GIT/sep-groupb
+ --output= /home/d/durmusy/Desktop/GIT/sep-groupb/slurm.%j.%N.out
  
Submit the job with needed args:

```bash

synthetic_job_script.sh

```
**Important**:

Do not change the args in the args.py, rather keep the project in line with remote/main
and give run the script above with different args:
For example:

+ xvfb-run -a python -u main.py
+ xvfb-run -a python -u main.py --no-tda
+ xvfb-run -a python -u main.py --crop = 2
+ ...
## **3. Checking the Job Status**

### **View All Jobs**

To check the status of your jobs:

```bash

squeue -u $USER

```

This will display your jobs in the queue, their status (`R` for running, `PD` for pending)
### **View Detailed Job Information**

To see more detailed information about a job (e.g., job 709481):

```bash

scontrol show job 709481

```

  

This will provide details like the job state, resource allocation, and reason for any delays (e.g., `Reason=Resources`).

  
## **4. Troubleshooting Pending Jobs**

  

### **Why Is My Job Pending?**

When a job is pending (`PD`), it could be waiting for:

- **Resources**: Requested resources (like GPUs) are currently unavailable.

- **Priority**: Other jobs have higher priority and are being scheduled first.

### **Check Resource Availability**

Run the following command to see available resources in the partition:

```bash

sinfo -p NvidiaAll

```

### **Check Queue for Pending Jobs**

You can see all pending jobs in the partition with:

```bash

squeue -p NvidiaAll --sort=S

```
## **5. Cancelling a Job**

If you want to cancel a job:

```bash

scancel <job_id>

```

For example, to cancel job 709481:

```bash

scancel 709481

```
