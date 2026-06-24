#!/bin/bash

echo "Submitting Node2Vec ablations"

# p
sbatch eval_n2v.sh 0.25 1 20 100 3
sbatch eval_n2v.sh 4 1 20 100 3

# q
sbatch eval_n2v.sh 1 0.25 20 100 3
sbatch eval_n2v.sh 1 4 20 100 3

# walks
sbatch eval_n2v.sh 1 1 10 100 3
sbatch eval_n2v.sh 1 1 40 100 3

# walk length
sbatch eval_n2v.sh 1 1 20 50 3
sbatch eval_n2v.sh 1 1 20 200 3

# window size
sbatch eval_n2v.sh 1 1 20 100 2
sbatch eval_n2v.sh 1 1 20 100 4

echo "Done"
