#!/bin/bash

# Baseline:
# hidden=128
# layers=1
# dropout=0.3
# lr=0.001

echo "Submitting GCN ablation jobs"

# Layers
sbatch gcn_ab.sh 128 1 0.3 0.001
sbatch gcn_ab.sh 128 3 0.3 0.001

# Hidden dimension
sbatch gcn_ab.sh 64 2 0.3 0.001
sbatch gcn_ab.sh 256 2 0.3 0.001

# Dropout
sbatch gcn_ab.sh 128 2 0.5 0.001
sbatch gcn_ab.sh 128 2 0.7 0.001

# Learning rate
sbatch gcn_ab.sh 128 2 0.3 0.01
sbatch gcn_ab.sh 128 2 0.3 0.1

echo "Done"
