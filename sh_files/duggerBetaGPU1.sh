#!/bin/bash

# FixMatch Justification CIFAR-10 (beta = 0.99)
./fixXperiment38.sh    # 

echo -e "For STL-10 100:100 beta = 0.9999" >> receipt28.txt
python3 analysis.py \
--out "experiments/stl10/FixMatch/result38_beta_0.9999_100_100" \
| tee -a receipt28.txt

./fixXperiment39.sh    # 

echo -e "For STL-10 100:100 beta = 0.9999" >> receipt28.txt
python3 analysis.py \
--out "experiments/stl10/FixMatch/result39_beta_0.9999_100_100" \
| tee -a receipt28.txt

./fixXperiment40.sh    # 

echo -e "For STL-10 100:100 beta = 0.9999" >> receipt28.txt
python3 analysis.py \
--out "experiments/stl10/FixMatch/result40_beta_0.9999_100_100" \
| tee -a receipt28.txt