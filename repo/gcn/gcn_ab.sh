#!/bin/bash

#SBATCH --job-name=EvalGCN3
#SBATCH --output=./logs/eval_gcn3_test_%j.out
#SBATCH --error=./logs/eval_gcn3_test_%j.err

#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Arguments
HID_DIM=$1
LAYERS=$2
DROPOUT=$3
LEARN_RATE=$4

# Variables
EMB_PATH="./ablation/gcn/gcn_emb_h${HID_DIM}_l${LAYERS}_d${DROPOUT}_lr${LEARN_RATE}.json"
TEST_PATH="./dataset/edges_2019.json"
OUTPUT_PATH="./ablation/gcn/results_h${HID_DIM}_l${LAYERS}_d${DROPOUT}_lr${LEARN_RATE}.json"
YEAR=2019
SCORE="dot"
NEG_SAMPLES=10000

# Your training commands here
hostname
date
echo "Running script now..."
singularity exec --nv ./python_3.14.sif python3 ./embedding_reader2.py \
	--embedding_path $EMB_PATH \
	--test_edge_path $TEST_PATH \
	--output_path $OUTPUT_PATH \
	--year $YEAR \
	--score_function $SCORE \
	--neg_samples $NEG_SAMPLES

echo "Finished job"
date

