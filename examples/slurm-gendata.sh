#!/bin/bash

#SBATCH --job-name=generate_data
#SBATCH --output=data-generation.out
#SBATCH --error=data-generation.err
#SBATCH --partition=general1
#SBATCH --time=01:00:00
#SBATCH --nodes=1

python ./examples/generate_dataset.py
