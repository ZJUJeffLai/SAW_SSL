#!/bin/bash

# CIFAR-10 (Table 1 Run 3)
./fixXperiment5.sh    # 50:50
./fixXperiment6.sh    # 100:100
./fixXperiment7.sh    # 150:150

echo -e "For CIFAR10 50:50" >> receipt.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch_True/result5_50_50" \
| tee -a receipt7.txt

echo -e "For CIFAR10 100:100" >> receipt.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch_True/result6_100_100" \
| tee -a receipt7.txt

echo -e "For CIFAR10 150:150" >> receipt.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch_True/result7_150_150" \
| tee -a receipt7.txt
