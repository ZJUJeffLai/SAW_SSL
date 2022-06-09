#!/bin/bash

# FixMatch Justification CIFAR-10 (beta = 0.9999)
./fixXperiment56.sh    # 


echo -e "For CIFAR10 (Pseudo-label) 100:100 beta = 0.9999" >> receipt35.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result53_beta_0.9999_pseudo" \
| tee -a receipt35.txt

./fixXperiment57.sh    # 

echo -e "For CIFAR10 (Pseudo-label) 100:100 beta = 0.9999" >> receipt35.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result54_beta_0.9999_pseudo" \
| tee -a receipt35.txt

./fixXperiment58.sh    # 

echo -e "For CIFAR10 (Pseudo-label) 100:100 beta = 0.9999" >> receipt35.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result55_beta_0.9999_pseudo" \
| tee -a receipt35.txt