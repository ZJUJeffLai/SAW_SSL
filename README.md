# SAW_SSL

This repository contains codes for the ICML 2022 paper
**"Smoothed Adaptive Weighting for Imbalanced Semi-Supervised Learning: Improve Reliability Against Unknown Distribution Data"** by Zhengfeng Lai, Chao Wang, Henrry Gunawan, Sen-Ching Cheung and Chen-Nee Chuah. 

The main contributers for these codes are Henrry Gunawan and Zhengfeng Lai. 

## Dependencies

* `python3`
* `pytorch == 1.1.0`
* `torchvision`
* `scipy`
* `randAugment (Pytorch re-implementation: https://github.com/ildoonet/pytorch-randaugment)`

## Scripts
We use "setup.sh" to set up the running environment on our server via Docker. But Conda is also working here as long as the above dependencies are set. 

For running different settings, please refer to "train_example.sh" as one example to set all parameters. 

Contact: lzhengfeng[at]ucdavis.edu
