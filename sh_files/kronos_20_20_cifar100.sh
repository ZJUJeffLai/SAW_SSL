#!/bin/bash

# CIFAR-100, 20:20
./remixXperiment17.sh
./remixXperiment18.sh
./remixXperiment19.sh

echo -e "For CIFAR100 20:20 (Turn 1)\n" >> receipt.txt
python3 analysis.py \
--out "experiments/cifar10/RemixMatch/result17_20_20" \
| tee -a receipt5.txt

echo -e "\nFor CIFAR100 20:20 (Turn 2) \n" >> receipt.txt
python3 analysis.py \
--out "experiments/cifar10/RemixMatch/result18_20_20" \
| tee -a receipt5.txt

echo -e "\nFor CIFAR100 20:20  (Turn 3) \n" >> receipt.txt
python3 analysis.py \
--out "experiments/cifar10/RemixMatch/result19_20_20" \
| tee -a receipt5.txt