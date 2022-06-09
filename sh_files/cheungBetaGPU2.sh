#!/bin/bash

# FixMatch Justification CIFAR-10 (beta = 0.999)
./fixXperiment29.sh    # 

echo -e "For CIFAR10 100:100 beta = 0.9999" >> receipt25.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result29_beta_0.9999_100_100" \
| tee -a receipt25.txt

./fixXperiment30.sh    # 

echo -e "For CIFAR10 100:100 beta = 0.9999" >> receipt25.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result30_beta_0.9999_100_100" \
| tee -a receipt25.txt

./fixXperiment31.sh    # 

echo -e "For CIFAR10 100:100 beta = 0.9999" >> receipt25.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result31_beta_0.9999_100_100" \
| tee -a receipt25.txt