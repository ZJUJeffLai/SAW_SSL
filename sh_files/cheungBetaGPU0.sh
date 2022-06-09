#!/bin/bash

# FixMatch Justification CIFAR-10 (beta = 0.99)
./fixXperiment23.sh    # 


echo -e "For CIFAR10 100:100 beta = 0.99" >> receipt23.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result23_beta_0.99_100_100" \
| tee -a receipt23.txt

./fixXperiment24.sh    # 

echo -e "For CIFAR10 100:100 beta = 0.99" >> receipt23.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result24_beta_0.99_100_100" \
| tee -a receipt23.txt

./fixXperiment25.sh    # 

echo -e "For CIFAR10 100:100 beta = 0.99" >> receipt23.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch/result25_beta_0.99_100_100" \
| tee -a receipt23.txt