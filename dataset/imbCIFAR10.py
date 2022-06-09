import numpy as np
from PIL import Image

import torchvision
import torch
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault

from .dataTools import createImbIdxs

# Parameters for data
cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

# Augmentations.
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

transform_strong = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])
transform_strong.transforms.insert(0, RandAugment(3, 4))
transform_strong.transforms.append(CutoutDefault(16))

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

transform_Tensor = transforms.Compose([transforms.ToTensor()])

class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3

def get_cifar10(root, l_samples, u_samples, test_samples, transform_train=transform_train, transform_strong=transform_strong,
                transform_val=transform_val, download=True):

    datasets = {}
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples)
    datasets["vanila"] = torchvision.datasets.CIFAR10(root, train=True, transform=transform_Tensor)

    # Generate Imbalanced Test Dataset
    base_dataset = torchvision.datasets.CIFAR10(root, train=False, download=download)
    imb_test_idxs = createImbIdxs(base_dataset.targets, test_samples)

    # Generate Reversed Imbalanced Test Dataset
    reversed_samples = test_samples[::-1]
    print("Reversed Test Samples: ", reversed_samples)
    reverse_test_idxs = createImbIdxs(base_dataset.targets, reversed_samples)

    # Training Datasets
    datasets["labeled"] = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
    datasets["unlabeled"] = CIFAR10_unlabeled(root, train_unlabeled_idxs, train=True,
                                                transform=TransformTwice(transform_train, transform_strong))

    # Test Datasets
    datasets["Test"] = CIFAR10_labeled(root, train=False, transform=transform_val, download=False)
    datasets["Imbalanced"] = CIFAR10_labeled(root, imb_test_idxs, train=False, transform=transform_val, download=False)
    datasets["Reversed"] = CIFAR10_labeled(root, reverse_test_idxs, train=False, transform=transform_val, download=False)

    # Fix Match Test Sets
    datasets["Weak"] = CIFAR10_labeled(root, train=False, transform=transform_train, download=False)
    datasets["Strong"] = CIFAR10_labeled(root, train=False, transform=transform_strong, download=False)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)}")
    return datasets
    # return train_labeled_dataset, train_unlabeled_dataset, test_dataset, imbalanced_dataset

def train_split(labels, n_labeled_per_class, n_unlabeled_per_class):
    num_class = len(n_labeled_per_class)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(num_class):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i] + n_unlabeled_per_class[i]])

    return train_labeled_idxs, train_unlabeled_idxs

# def createImbIdxs(labels, n_data_per_class) :
#     '''
#     Creates a List containing Indexes of the Imbalanced Classification

#     Input: 
#         labels: Ground Truth of Dataset
#         n_data_per_class: Class Distribution of Dataset desired

#     Output:
#         data_idxs: List containing indexes for Dataset 
#     '''
#     labels = np.array(labels) # Classification Ground Truth 
#     data_idxs = []  # Collect Ground Truth Indexes

#     for i in range( len(n_data_per_class) ) :
#         idxs = np.where(labels == i)[0]
#         data_idxs.extend(idxs[ :n_data_per_class[i] ])

#     return data_idxs


class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    

class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])

if __name__ == '__main__':
    # Example
    from dataTools import make_imb_data

    num_max = 150   # Max Number of Dataset per class
    num_class = 100 # CIFAR-100
    imb_ratio_l = 100   # Labeled Class Imbalance Ratio
    imb_ratio_u = 100   # Unlabeled Class Imbalance Ratio
    ratio = 2.0 # Relative Size between Labeled and Unlabeled

    N_SAMPLES_PER_CLASS = make_imb_data(num_max, num_class, imb_ratio_l)  # ground_truth labeled
    U_SAMPLES_PER_CLASS = make_imb_data(ratio * num_max, num_class, imb_ratio_u) # ground truth unlabeled
    IMB_TEST_PER_CLASS = make_imb_data(1000, num_class, imb_ratio_u) # test dataset

    datasets = get_cifar10('/root/data', N_SAMPLES_PER_CLASS,
                                                        U_SAMPLES_PER_CLASS, IMB_TEST_PER_CLASS)

    print("Dictionary of Datasets", datasets)

    import sys
    sys.path.append('/root/imbalancedSSL/')
    from utils.misc import get_mean_and_std
    mean, std = get_mean_and_std(datasets["vanila"])

    print("Mean: ", mean)
    print("STD : ", std)
