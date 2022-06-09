#!/bin/bash

# CIFAR100 10:10
./remixXperiment14.sh
./remixXperiment15.sh
./remixXperiment16.sh

echo -e "For CIFAR100 10:10 (Turn 1)\n" >> receipt.txt
python3 analysis.py \
--out "experiments/cifar10/RemixMatch/result14_10_10" \
| tee -a receipt4.txt

echo -e "\nFor CIFAR100 10:10 (Turn 2) \n" >> receipt.txt
python3 analysis.py \
--out "experiments/cifar10/RemixMatch/result15_10_10" \
| tee -a receipt4.txt

echo -e "\nFor CIFAR100 10:10  (Turn 3) \n" >> receipt.txt
python3 analysis.py \
--out "experiments/cifar10/RemixMatch/result16_10_10" \
| tee -a receipt4.txt