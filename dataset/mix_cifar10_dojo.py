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

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2= self.transform(inp)
        return out1, out2

def get_cifar10(root, l_samples, u_samples, test_samples, transform_train=transform_train, transform_strong=transform_strong,
                transform_val=transform_val, download=True):

    datasets = {}
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples)

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
                                                transform=TransformTwice(transform_train))

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
