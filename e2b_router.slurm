#!/bin/bash

#SBATCH --partition=cpu-ferranti
#SBATCH --mem=128g
#SBATCH --cpus-per-task=16
#SBATCH --output=./logs/e2b_router/%x-%j.out
#SBATCH --error=./logs/e2b_router/%x-%j.err
#SBATCH --time=3-00:00
#SBATCH --mail-type=ALL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=haoyu.he@uni-tuebingen.de  # Email to which notifications will be sent

# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID
echo -e "---------------------------------\n"

# Due to a potential bug, we need to manually load our bash configurations first
source $HOME/.bashrc
conda activate dlm
# Run our code
echo "-------- PYTHON OUTPUT ----------"
cd /home/geiger/gwb130/mdpo/
srun python e2b_router.py
echo "---------------------------------"

# Deactivate environment again
conda deactivate