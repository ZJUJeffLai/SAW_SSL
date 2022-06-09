#!/bin/bash

# FixMatch Justification CIFAR-10 (beta = 0.99)
./fixXperiment35.sh    # 


echo -e "For STL-10 100:100 beta = 0.999" >> receipt27.txt
python3 analysis.py \
--out "experiments/stl10/FixMatch/result35_beta_0.999_100_100" \
| tee -a receipt27.txt

./fixXperiment36.sh    # 

echo -e "For STL-10 100:100 beta = 0.999" >> receipt27.txt
python3 analysis.py \
--out "experiments/stl10/FixMatch/result36_beta_0.999_100_100" \
| tee -a receipt27.txt

./fixXperiment37.sh    # 

echo -e "For STL-10 100:100 beta = 0.999" >> receipt27.txt
python3 analysis.py \
--out "experiments/stl10/FixMatch/result37_beta_0.999_100_100" \
| tee -a receipt27.txt