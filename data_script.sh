#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --account=def-khalile2
#SBATCH --mem=32000M        # memory per node

python pipeline.py