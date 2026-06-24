#!/bin/bash

#SBATCH --job-name=N2Vab
#SBATCH --output=./logs/n2v_ab_%j.out
#SBATCH --error=./logs/n2v_ab_%j.err

#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

P=$1
Q=$2
WALKS=$3
LENGTH=$4
WINDOW_SIZE=$5
OUTPUT="./ablation/n2v/embeddings_p${P}_q${Q}_w${WALKS}_l${LENGTH}_ws${WINDOW_SIZE}.txt"

# Your training commands here
hostname
date
echo "Running script now..."

singularity exec --nv ./python_3.11.sif python3 ./n2v.py \
    --graph_path "./dataset/edges_existed_sumweight.json" \
    --output_path $OUTPUT \
    --emb_dim 128 \
    --window_size $WINDOW_SIZE \
    --neg_samples 10 \
    --batch_size 1024 \
    --epochs 10 \
    --lr 0.01 \
    --walks $WALKS \
    --length $LENGTH \
    --workers 32 \
    --p $P \
    --q $Q

echo "Finished Job"
date
