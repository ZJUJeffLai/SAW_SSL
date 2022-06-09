#!/bin/bash

# FixMatch Justification CIFAR-10 (beta = 0.999)
./fixXperiment53.sh    # 


echo -e "For CIFAR10 (Pseudo-label) 100:100 beta = 0.999" >> receipt35.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result53_beta_0.999_pseudo" \
| tee -a receipt35.txt

./fixXperiment54.sh    # 

echo -e "For CIFAR10 (Pseudo-label) 100:100 beta = 0.999" >> receipt35.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result54_beta_0.999_pseudo" \
| tee -a receipt35.txt

./fixXperiment55.sh    # 

echo -e "For CIFAR10 (Pseudo-label) 100:100 beta = 0.999" >> receipt35.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result55_beta_0.999_pseudo" \
| tee -a receipt35.txt