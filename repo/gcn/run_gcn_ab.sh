#!/bin/bash

#SBATCH --job-name=GCN
#SBATCH --output=./logs/gcn_ab%j.out
#SBATCH --error=./logs/gcn_ab%j.err

#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Arguments
HID_DIM=$1
LAYERS=$2
DROPOUT=$3
LEARN_RATE=$4

# Variables
INPUT_PATH="./dataset/combined.json"
OUTPUT_PATH="./ablation/gcn/gcn_emb_h${HID_DIM}_l${LAYERS}_d${DROPOUT}_lr${LEARN_RATE}.json"
SPLIT_YEAR=2018
EMB_DIM=128
EPOCHS=100
DECAY=1e-5

# Your training commands here
hostname
date
echo "Running script now..."
singularity exec --nv ./python_3.14.sif python3 ./gcn.py \
	--data_path $INPUT_PATH\
	--results_path $OUTPUT_PATH\
	--split_year $SPLIT_YEAR\
	--hidden_dim $HID_DIM\
	--embedding_dim $EMB_DIM\
	--layers $LAYERS\
	--epochs $EPOCHS\
	--learn_rate $LEARN_RATE\
	--dropout $DROPOUT\
	--decay $DECAY

echo "Finished job"
date
