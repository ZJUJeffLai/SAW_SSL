#!/bin/bash

# FixMatch Justification CIFAR-10 (beta = 0.999)
./fixXperiment26.sh    # 

echo -e "For CIFAR10 100:100 beta = 0.999" >> receipt24.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result26_beta_0.999_100_100" \
| tee -a receipt24.txt

./fixXperiment27.sh    # 

echo -e "For CIFAR10 100:100 beta = 0.999" >> receipt24.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result27_beta_0.999_100_100" \
| tee -a receipt24.txt

./fixXperiment28.sh    # 

echo -e "For CIFAR10 100:100 beta = 0.999" >> receipt24.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result28_beta_0.999_100_100" \
| tee -a receipt24.txt