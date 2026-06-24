#!/bin/bash

#SBATCH --job-name=Spec2048
#SBATCH --output=./logs/spec_emb_2048_%j.out
#SBATCH --error=./logs/spec_emb_2048_%j.err

#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Variables
JSON_PATH="./dataset/combined.json"
VAL_YEAR=2018
TEST_YEAR=2019
EMB_DIM=2048
TOPK=50
OUTPUT_DIR="./ablation/specemb"
NEG_SAMPLES=10000

# Your training commands here
hostname
date
echo "Running script now..."
singularity exec --nv ./python_3.14.sif python3 ./specemb.py \
	--json_path $JSON_PATH \
	--val_year $VAL_YEAR \
	--test_year $TEST_YEAR \
	--embed_dim $EMB_DIM \
	--top_k $TOPK \
	--output_dir $OUTPUT_DIR \
	--neg_sample $NEG_SAMPLES

echo "Finished job"
date

