
#from torchsampler import ImbalancedDatasetSampler

import numpy as np  # For createImbIdxs() & make_imb_data()
import torch    # For gtDict() & prob2Distb()
import torch.utils.data as data # For prepareDataLoaders()

'''
Contains Tools for Datasets
'''

def createImbIdxs(labels, n_data_per_class) :
    '''
    Creates a List containing Indexes of the Imbalanced Classification

    Input: 
        labels: Ground Truth of Dataset
        n_data_per_class: Class Distribution of Dataset desired

    Output:
        data_idxs: List containing indexes for Dataset 
    '''
    labels = np.array(labels) # Classification Ground Truth 
    data_idxs = []  # Collect Ground Truth Indexes

    for i in range( len(n_data_per_class) ) :
        idxs = np.where(labels == i)[0]
        data_idxs.extend(idxs[ :n_data_per_class[i] ])

    return data_idxs

def checkReverseDistb(imb_ratio) :
    reverse = False
    if imb_ratio / abs(imb_ratio) == -1 :
        reverse = True
        imb_ratio = imb_ratio * -1

    return reverse, imb_ratio

def make_imb_data(max_num, class_num, gamma):
    reverse, gamma = checkReverseDistb(gamma)

    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    if reverse :
        class_num_list.reverse()
    print(class_num_list)
    return list(class_num_list)

def gtDict(labeled, unlabeled, use_cuda) :
    '''
    Returns Dictionary containing Class Distributions

    Inputs: 
        labeled: Labeled Class Distribution (List, NumPy, PyTorch, etc)
        unlabeled: Unlabeled Class Distribution
    '''
    # Collect Ground Truth of Unsupervised Class Distribution 
    gt_distribution_l = torch.Tensor(labeled)
    gt_distribution_u = torch.Tensor(unlabeled)
    if use_cuda :
        gt_distribution_l = gt_distribution_l.cuda()
        gt_distribution_u = gt_distribution_u.cuda()
    
    gt_distb = gt_distribution_l + gt_distribution_u

    # Dictionary containing Class Distribution
    distb_dict = {"gt"    : gt_distb,
                    "gt_l"  : gt_distribution_l,
                    "gt_u"  : gt_distribution_u} 

    return distb_dict

def prepareDataLoaders(datasets, batch_size) :
    dataLoaders = {}
    dataLoaders["labeled"] = data.DataLoader(datasets["labeled"], 
#                                              sampler=ImbalancedDatasetSampler(datasets["labeled"]),         
                                             batch_size=batch_size, 
                                             shuffle=True, num_workers=4, drop_last=True)

    dataLoaders["unlabeled"] = data.DataLoader(datasets["unlabeled"], batch_size=batch_size, shuffle=True, num_workers=4,
                                            drop_last=True)
    dataLoaders["Test"] = data.DataLoader(datasets["Test"], batch_size=batch_size, shuffle=False, num_workers=4)
    dataLoaders["Imbalanced"] = data.DataLoader(datasets["Imbalanced"], batch_size=batch_size, shuffle=False, num_workers=4)
    dataLoaders["Reversed"] = data.DataLoader(datasets["Reversed"], batch_size=batch_size, shuffle=False, num_workers=4)
    dataLoaders["Weak"] = data.DataLoader(datasets["Weak"], batch_size=batch_size, shuffle=False, num_workers=4)
    dataLoaders["Strong"] = data.DataLoader(datasets["Strong"], batch_size=batch_size, shuffle=False, num_workers=4)

    return dataLoaders

def prob2Distribution(confidence, use_cuda):
    '''
    Converts Probability Output of the Model
    into Classification Distribution through
    Summing One-Hot Encoding
    
    Input:
    confidence : torch Array containing Probability
                  output of the model
                 Assumes Array Shape (# of data, num_class)

    use_cuda   : Check if Using CUDA
    
    Output:
    classDistribution: torch Array containg Class Distribution
    '''
    maxValues, classIndex = torch.max(confidence, dim=1)
    size = classIndex.shape[0]
    num_class = confidence.shape[1] 

    # Transform label to one-hot
    if use_cuda :
        classIndex = torch.zeros(size, \
            num_class).cuda().scatter_(1, classIndex.view(-1, 1), 1) # torch.Size([11163, 10])
    else :
        classIndex = torch.zeros(size, \
            num_class).scatter_(1, classIndex.view(-1, 1), 1) # torch.Size([11163, 10])
    classDistribution = torch.sum(classIndex, dim=0)       # torch.Size([10])

    return classDistribution
