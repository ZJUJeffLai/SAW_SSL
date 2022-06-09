#!/bin/bash

# FixMatch Justification CIFAR-10 (beta = 0.99)
./fixXperiment32.sh    # 

echo -e "For STL-10 100:100 beta = 0.99" >> receipt26.txt
python3 analysis.py \
--out "experiments/stl10/FixMatch/result32_beta_0.99_100_100" \
| tee -a receipt26.txt

./fixXperiment33.sh    # 

echo -e "For STL-10 100:100 beta = 0.99" >> receipt26.txt
python3 analysis.py \
--out "experiments/stl10/FixMatch/result33_beta_0.99_100_100" \
| tee -a receipt26.txt

./fixXperiment34.sh    # 

echo -e "For STL-10 100:100 beta = 0.99" >> receipt26.txt
python3 analysis.py \
--out "experiments/stl10/FixMatch/result34_beta_0.99_100_100" \
| tee -a receipt26.txt
