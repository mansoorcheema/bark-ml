#!/bin/bash
#SBATCH --qos normal
#SBATCH -c 2
##SBATCH --gres=gpu:1
#SBATCH --mem=10GB

srun singularity exec --nv ../images/bark_ml.img python3 -u ./configuration 
