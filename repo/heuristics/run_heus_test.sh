#!/bin/bash

#SBATCH --job-name=HeurTest
#SBATCH --output=./logs/heuristics_test2_%j.out
#SBATCH --error=./logs/heuristics_test2_%j.err

#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Variables
INPUT_PATH="./dataset/combined.json"
OUTPUT_DIR="./results"
TRAIN_YEAR=2017
VALIDATION_YEAR=2018
TEST_YEAR=2019
NEG_SAMPLES=10000
HITK=50

# Your training commands here
hostname
date
echo "Running script now..."
singularity exec --nv ./python_3.14.sif python3 ./heuristics_test.py \
	--data_path $INPUT_PATH \
	--results_dir $OUTPUT_DIR \
	--train_end_year $TRAIN_YEAR \
	--valid_year $VALIDATION_YEAR \
	--test_year $TEST_YEAR \
	--negative_samples $NEG_SAMPLES \
	--hit_k $HITK \

echo "Finished job"
date
