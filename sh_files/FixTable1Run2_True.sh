#!/bin/bash

# CIFAR-10 (Table 1 Run 2)
./fixXperiment2.sh    # 50:50
./fixXperiment3.sh    # 100:100
./fixXperiment4.sh    # 150:150

echo -e "For CIFAR10 50:50" >> receipt.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch_True/result2_50_50" \
| tee -a receipt6.txt

echo -e "For CIFAR10 100:100" >> receipt.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch_True/result3_100_100" \
| tee -a receipt6.txt

echo -e "For CIFAR10 150:150" >> receipt.txt
python3 analysis.py \
--out "experiments/cifar10/FixMatch_True/result4_150_150" \
| tee -a receipt6.txt