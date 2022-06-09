#!/bin/bash

# FixMatch Justification CIFAR-10 (beta = 0.99)
./fixXperiment50.sh    # 


echo -e "For CIFAR10 (Pseudo-label) 100:100 beta = 0.99" >> receipt35.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result50_beta_0.99_pseudo" \
| tee -a receipt35.txt

./fixXperiment51.sh    # 

echo -e "For CIFAR10 (Pseudo-label) 100:100 beta = 0.99" >> receipt35.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result51_beta_0.99_pseudo" \
| tee -a receipt35.txt

./fixXperiment52.sh    # 

echo -e "For CIFAR10 (Pseudo-label) 100:100 beta = 0.99" >> receipt35.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result52_beta_0.99_pseudo" \
| tee -a receipt35.txt