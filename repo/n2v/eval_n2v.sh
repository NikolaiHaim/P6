#!/bin/bash

#SBATCH --job-name=N2VEVAL
#SBATCH --output=./logs/n2v_eval_%j.out
#SBATCH --error=./logs/n2v_eval_%j.err

#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:0

P=$1
Q=$2
WALKS=$3
LENGTH=$4
WINDOW_SIZE=$5

EMBEDDING="./ablation/n2v/embeddings_p${P}_q${Q}_w${WALKS}_l${LENGTH}_ws${WINDOW_SIZE}.txt"
OUTPUT="./ablation/n2v/results_p${P}_q${Q}_w${WALKS}_l${LENGTH}_ws${WINDOW_SIZE}.json"

TEST_PATH="./dataset/edges_2019.json"

# Your training commands here
hostname
date
echo "Running script now..."

singularity exec --nv ./python_3.14.sif python3 ./embedding_reader3.py \
    --embedding_path $EMBEDDING \
    --test_edge_path $TEST_PATH \
    --output_path $OUTPUT \
    --year 2019 \
    --score_function "dot" \
    --neg_samples 10000

echo "Finished Job"
date
