#!/bin/bash
# created: Nov 30, 2018 2:51 PM
# author: mayrajan
# Example script to run batch job in csc taito-gpu
# arguments in order
# -o output stream file
# -e error stream file
# -p partition to use
# --nodes nodes to use
# --gres general resources. here 1 p100 gpu.
# --cpus-per-task number of cpus. here 2 cores
# --mem reserved memory. here 12000 MB
# -t time to use. here 12 hours
#SBATCH -J LULC_UNET_training
#SBATCH -o oversample_grass_ce_multi_unet.out
#SBATCH -e oversample_grass_ce_multi_unet.err
#SBATCH -p gpu 
#SBATCH --nodes=1 
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000 
#SBATCH -t 12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=add.your@own.mail

# note, this job requests a total of 2 cores and 1 GPGPU cards
# note, submit the job from taito-gpu.csc.fi
# commands to manage the batch script
#   submission command
#     sbatch [script-file]
#   status command
#     squeue -u mayrajan
#   termination command
#     scancel [jobid]

# For more information
#   man sbatch
#   more examples in Taito GPU guide in
#   http://research.csc.fi/taito-gpu

# example run commands
echo "$(date)"
# load python module
module purge
module load python-env/3.5.3-ml
# Move to work directory
cd $WRKDIR/lulc_ml
python train_model.py 
echo "$(date)"
# This script will print some usage statistics to the
# end of file: multiclass_unet.out
# Use that to improve your resource request estimate
# on later jobs.
seff $SLURM_JOBID
