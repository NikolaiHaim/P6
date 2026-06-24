#!/bin/bash

#SBATCH --job-name=SVD_AB
#SBATCH --output=./logs/svd_ab_%j.out
#SBATCH --error=./logs/svd_ab_%j.err

#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Variables
INPUT_PATH="./dataset/combined.json"
OUTPUT_DIR="./ablation/svd"
TRAIN_YEAR=2017
VALIDATION_YEAR=2018
TEST_YEAR=2019
EMB_DIM=$1
HITK=50
NEG_SAMPLES=10000

# Your training commands here
hostname
date
echo "Running script now..."
singularity exec --nv ./python_3.14.sif python3 ./svd.py \
	--input_path $INPUT_PATH \
	--output_dir $OUTPUT_DIR \
	--train_year $TRAIN_YEAR \
	--valid_year $VALIDATION_YEAR \
	--test_year $TEST_YEAR \
	--embedding_dim $EMB_DIM \
	--hit_k $HITK \
	--negative_ratio $NEG_SAMPLES 

echo "Finished Job"
date
