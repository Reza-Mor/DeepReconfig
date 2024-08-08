#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --account=def-khalile2
#SBATCH --mem=32000M        # memory per node

# 30, 70, 110, 150 nodes
# 0.35, 0.65, 0.85 avg link utilization
# sizes=(30, 70, 110, 150)
# utilizations=(0.35, 0.55, 0.75)

i=1
for size in {30,60}; do
    for util in {0.15,0.30,0.45,0.6,0.75,0.9}; do
        echo $size $util
        python generate_flow_dataset.py --output_file "datasets/flows/dataset_$i" --avg_link_utilization $util --graph_size 180
        python Flows_MIP.py --dataset_file "datasets/flows/dataset_$i" --minimize_rounds True
        i=$((i+1))
    done
done
