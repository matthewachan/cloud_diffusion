#!/bin/bash
# The line above this is the "shebang" line.  It must be first line in script
#-----------------------------------------------------
#	Default OnDemand Job Template
#	For a basic Hello World sequential job
#-----------------------------------------------------
#
# Slurm sbatch parameters section:
#	Request walltime
#SBATCH --time=12:00:00
#	Request 1 GB of memory per CPU core
#SBATCH --mem-per-cpu=1gb
#	Allow other jobs to run on same node
#SBATCH --oversubscribe
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -o /fs/nexus-scratch/mattchan/hrrr-diffusion/logs/slurm-%j.txt

# This job will run our hello-umd demo binary in sequential mode
# Output will go to local /tmp scratch space on the node we are running
# on, and then will be copied back to our work directory.

# Section to make a scratch directory for this job
# For sequential jobs, local /tmp filesystem is a good choice
# We include the SLURM jobid in the #directory name to avoid interference if
# multiple jobs running at same time.
TMPWORKDIR="/tmp/ood-job.${SLURM_JOBID}"
mkdir $TMPWORKDIR
cd $TMPWORKDIR

# Section to output information identifying the job, etc.
echo "Slurm job ${SLURM_JOBID} running on $(hostname)"
echo "To run on ${SLURM_NTASKS} CPU cores across ${SLURM_JOB_NUM_NODES} nodes"
echo "Threads: $(nproc), GPU IDs: ${CUDA_VISIBLE_DEVICES} ($(nvidia-smi --query-gpu=gpu_name --format=csv,noheader))"
free -h

echo "All nodes: ${SLURM_JOB_NODELIST}"
date
pwd
echo "Loaded modules are:"
module list

python /fs/nexus-scratch/mattchan/hrrr-diffusion/scripts/dl_era5.py
# python /fs/nexus-scratch/mattchan/hrrr-diffusion/train_uvit.py --epochs 200

# Save the exit code from the previous command
ECODE=$?

echo "Job finished with exit code $ECODE"
date

# Exit with the cached exit code
exit $ECODE
