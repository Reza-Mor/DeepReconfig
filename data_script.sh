#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --account=def-someuser
#SBATCH --mem=32000M        # memory per node

python pipeline.py