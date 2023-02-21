#!/bin/bash
#SBATCH --account=def-khalile2
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=2   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=03:00:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

#module load cuda cudnn 
#source tensorflow/bin/activate
#python train_policy.py dataset1_20by20 Rnaenv_v1 ff1 500
#python train_policy.py dataset1_20by20 Rnaenv_v1 ff1 500
#python train_policy.py dataset1_20by20 Rnaenv_v1 ff1 500
python train_policy.py --dataset dataset1_20by20 --environment Rnaenv_v1 --model_config ff1 --n_iter 3 --gamma 0.99
