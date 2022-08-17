#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=sar_cnn_classification
#SBATCH --ntasks=1
#SBATCH --output=result.log
#SBATCH --mem=0
#SBATCH --time=0
$HOME/python_envs/tensorflow1.15/bin/python3 main.py --test_dir=/content/test_synthetic > result.out
