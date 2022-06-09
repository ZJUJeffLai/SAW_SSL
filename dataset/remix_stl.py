import numpy as np
from PIL import Image

import torchvision
import torch
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault

stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2471, 0.2435, 0.2616)

# Augmentations.
transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(stl10_mean, stl10_std)
    ])

transform_strong = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(stl10_mean, stl10_std)
])
transform_strong.transforms.insert(0, RandAugment(3, 4))
transform_strong.transforms.append(CutoutDefault(16))

transform_val = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(stl10_mean, stl10_std)
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

class TransformMM:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_stl10(root, l_samples, method = 'remix', transform_train=transform_train, transform_strong=transform_strong,
                transform_val=transform_val, download=True):

    datasets = {}
    base_dataset = torchvision.datasets.STL10(root, split='train', download=download)
    datasets["vanila"] = torchvision.datasets.STL10(root, split='train+unlabeled', transform=transform_Tensor)

    ##### Labeled data
    train_labeled_idxs = train_split(base_dataset.labels, l_samples)

    # For the efficiency (duplication of selected indices)
    for i in range(2):
        train_labeled_idxs.extend(train_labeled_idxs)

    if method == 'remix':
        train_labeled_dataset = STL10_labeled(root, train_labeled_idxs, split='train', transform=transform_strong)
    else:
        train_labeled_dataset = STL10_labeled(root, train_labeled_idxs, split='train', transform=transform_train)

    ##### Unlabeled data
    if method == 'mix':
        train_unlabeled_dataset = STL10_unlabeled(root, indexs=None, split='unlabeled',
                                                    transform=TransformMM(transform_train))
    elif method == 'fix':
        labeled_data = base_dataset.data[train_labeled_idxs]
        train_unlabeled_dataset = STL10_unlabeled(root, indexs=None, split='unlabeled',
                                                    transform=TransformTwice(transform_train, transform_strong), added_data=labeled_data)
    else:
        train_unlabeled_dataset = STL10_unlabeled(root, indexs=None, split='unlabeled',
                                                    transform=TransformTwice(transform_train, transform_strong))

    test_dataset = STL10_labeled(root, split='test', transform=transform_val, download=False)
    
    # My Dictionary
    datasets["labeled"] = train_labeled_dataset
    datasets["unlabeled"] = train_unlabeled_dataset
    datasets["test"] = test_dataset

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_dataset.data)}")
    # return train_labeled_dataset, train_unlabeled_dataset, test_dataset
    return datasets

def train_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    num_class = len(n_labeled_per_class)

    for i in range(num_class):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])

    return train_labeled_idxs

class STL10_labeled(torchvision.datasets.STL10):

    def __init__(self, root, indexs=None, split='train',
                 transform=None, target_transform=None,
                 download=False, added_data=None):
        super(STL10_labeled, self).__init__(root, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]

        if added_data is not None:
            self.data = np.concatenate((self.data, added_data), axis=0)
            self.labels = np.concatenate((self.labels, self.labels[:len(added_data)]), axis=0)

        self.data = [Image.fromarray(np.transpose(img, (1, 2, 0))) for img in self.data]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

class STL10_unlabeled(torchvision.datasets.STL10):
    def __init__(self, root, indexs, split='unlabeled',
                 transform=None, target_transform=None,
                 download=False, added_data=None):
        super(STL10_unlabeled, self).__init__(root, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)

        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array([-1 for i in range(len(self.labels))])

        if added_data is not None:
            self.data = np.concatenate((self.data, added_data), axis=0)
            self.labels = np.concatenate((self.labels, self.labels[:len(added_data)]), axis=0)

        self.data = [Image.fromarray(np.transpose(img, (1, 2, 0))) for img in self.data]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    # Example
    from dataTools import make_imb_data
    from dataTools import createImbIdxs

    num_max = 150   # Max Number of Dataset per class
    num_class = 10 # CIFAR-100
    imb_ratio_l = 100   # Labeled Class Imbalance Ratio
    imb_ratio_u = 100   # Unlabeled Class Imbalance Ratio
    ratio = 2.0 # Relative Size between Labeled and Unlabeled

    N_SAMPLES_PER_CLASS = make_imb_data(num_max, num_class, imb_ratio_l)  # ground_truth labeled
    U_SAMPLES_PER_CLASS = make_imb_data(ratio * num_max, num_class, imb_ratio_u) # ground truth unlabeled
    IMB_TEST_PER_CLASS = make_imb_data(1000, num_class, imb_ratio_u) # test dataset

    datasets = get_stl10('/root/data', N_SAMPLES_PER_CLASS,
                                                        U_SAMPLES_PER_CLASS, IMB_TEST_PER_CLASS)

    print("Dictionary of Datasets", datasets)

    import sys
    sys.path.append('/root/imbalancedSSL/')
    from utils.misc import get_mean_and_std
    mean, std = get_mean_and_std(datasets["vanila"])

    print("Mean: ", mean)
    print("STD : ", std)
