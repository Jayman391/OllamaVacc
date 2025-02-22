#!/bin/bash

#####  Slurm preamble
# Job name is descriptive and optional

#SBATCH --job-name=llama_label
#SBATCH --partition=nvgpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=5gb
#SBATCH --time=10:00:00

# Job setup and confirmation
module purge
module load apptainer
my_job_header

# Work the job should do

# go to the directory where the data lives
cd $HOME/ollama

# Do NOT forget the --nv or no GPU will be accessible
apptainer exec --nv /gpfs1/cont/ollama/full-ollama-0.5.7.sif ./run_requests.sh

