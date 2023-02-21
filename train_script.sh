#!/bin/bash
#SBATCH --account=def-khalile2
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=3  # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G        # memory per node
#SBATCH --time=07:00:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

#module load cuda cudnn 
#source tensorflow/bin/activate
#python train_policy.py dataset1_20by20 Rnaenv_v1 ff1 500
#python train_policy.py dataset1_20by20 Rnaenv_v1 ff1 500
#python train_policy.py dataset1_20by20 Rnaenv_v1 ff1 500
python train_policy.py --dataset dataset1_20by20 --environment Rnaenv_v5 --model_config ff2 --n_iter 5000 --gamma 0.99
