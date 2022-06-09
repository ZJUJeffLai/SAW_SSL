import torch
from copy import deepcopy

# import argparse   # For parseClassWeights

def parseClassWeights(parser) :
        # Weights for Model's Cost Function
    # parser.add_argument('--w_L', choices=["", "default", "total", "minority"], help='Applying Weights to Loss: \
    #     \n (default/blank) = Uniform Weight of Ones \
    #     \n total = Class Distribution / Total Class Distribution : [1, 3] \
    #     \n minority = Class Distribution / Minority Class Distribution : [1, inf]') # Old Arugment
    parser.add_argument('--distbu', choices=["", "uniform", "pseudo", \
        "weak", "strong", "gt", "gt_l", "gt_u"], \
        help='Applying Weights to Unsupervised Loss \
        \n (blank/uniform) = Uniform Weight of Ones \
        \n pseudo = Using Pseudo-Label Class Distribution \
        \n weak = Using Weakly Augmented Output Class Distribution \
        \n strong = Using Strongly Augmented Output Class Distribution \
        \n gt = Using Ground Truth Class Distribution (Labeled + Unlabeled) \
        \n gt_l = Using Ground Truth Class Distribution (Labeled) \
        \n gt_u = Using Ground Truth Class Distribution (Unlabeled)') 

    parser.add_argument('--distbl', choices=["", "uniform", "gt_l"], \
        help="Applying Weights to Supervised Loss \
            \n (blank/uniform) = Uniform Weight of Ones \
            \n gt_l = Using Ground Truth Class Distribution (Labeled)")

    # For Weighting Function Schemes
    parser.add_argument('--invert', action='store_true', \
        help='If declared, flip class weights on Loss (Penalize Minority more than Majority)')
    parser.add_argument('--normalize', default=None, type=float, \
        help='Normalize class weights on Loss according to number of classes \
            \n such that sum(weights) = num_class * norm_const')
    parser.add_argument('--total', default=None, type=float, \
        help='Using Total-Schemed Weights to Unsupervised Loss m*(Class/Total) + 1')
    parser.add_argument('--minority', default=None, type=float, \
        help='Using Minority-Schemed Weights to Unsupervised Loss (Class/Minority)')
    parser.add_argument('--intercept', default=None, type=float, \
        help='Using Intercept-Schemed Weights to Unsupervised Loss (Class/Total) + b')
    parser.add_argument('--log', default=None, type=float, \
        help='Using Minority-Schemed Weights to Unsupervised Loss (log(a*Class)/log(Total))')
    parser.add_argument('--effective', default=None, type=float, \
        help='Using Effective Number-Schemed Weights to Unsupervised Loss ((1-beta)/(1-beta^Class)) \
            \n Note: Hyperparameter is automatically calculated')
    parser.add_argument('--power', default=None, type=float, \
        help='Using Powered-Schemed Weights to Unsupervised Loss (Total/Class)^alpha')

    return parser

# Original Function Definition
# def createSettings(num_class, use_cuda, \
#     distbu=args.distbu, distbl=args.distbl, \
#     invert=args.invert, normalize=args.normalize, \
#      total=args.total, minority=args.minority,  \
#          intercept=args.intercept, log=args.log, \
#              effective=args.effective, power=args.power) :

def createSettings(num_class, use_cuda, \
    distbu, distbl, \
    invert, normalize, \
     total, minority,  \
         intercept, log, \
             effective, power) :
    # Settings for Weighted loss based on Class Distribution (for Unsupervised)
    class_weight_u = torch.ones(num_class)
    if use_cuda :
        class_weight_u = class_weight_u.cuda()
    
    distbLoss_dict = {"unlabeled" : distbu, 
                    "labeled" : distbl}

    if (total != None and minority != None) :
        raise Exception("Both --total and --minority declared \n \
            Please Choose Only one of them \n \
                Exiting ...")
    elif (total != None) :
        weightLoss = {"type": "total",
                        "const": total,
                        "invert?" : invert,
                        "normalize" : normalize}
    elif (minority != None) :
        weightLoss = {"type": "minority",
                        "const": minority,
                        "invert?" : invert,
                        "normalize" : normalize}
    elif (intercept != None) :
      weightLoss = {"type": "intercept",
                        "const": intercept,
                        "invert?" : invert,
                        "normalize" : normalize} # Feature to be Fixed
    elif (log != None) :
      weightLoss = {"type": "log",
                        "const": log,
                        "invert?" : invert,
                        "normalize" : normalize} # Feature to be Fixed
    elif (effective != None) :
        weightLoss = {"type": "effective",
                        "const": effective,
                        "invert?" : invert,
                        "normalize" : normalize} # Feature to be Fixed
    elif (power != None) :
        weightLoss = {"type": "power",
                        "const": power,
                        "invert?" : invert,
                        "normalize" : normalize} # Feature to be Fixed
    else :
        weightLoss = None

    return class_weight_u, distbLoss_dict, weightLoss

def invertDistribution(vec) :
    '''
    Returns the inverted Distribution
    based on ascending order (inverted Tail)
    (Class Minority -> Class Majority)

    Inputs:
        vec : PyTorch Tensor (Shaped as a Vector)
              that contains the Class Distribution

    Outputs:
        vec : PyTorch Tensor containing Ascending
                Distribution
    
    Note: Output is not strictly sorted, but arranged
        such that the Distribution are inverted to
        preserve the class structure
    '''
    values, indexes = torch.sort(vec)   # To check Ascending Distribution Values
    L = len(vec)

    for idx in range( ceil(L/2) ):
        temp = deepcopy( vec[ indexes[idx] ] )
        vec[indexes[idx]] = vec[ indexes[L-idx-1] ]
        vec[ indexes[L-idx-1] ] = temp

    return vec

def getClassWeights(distbLoss, weightLoss, epoch, darp, distb_dict, use_cuda=False) :
    '''
    Calculate Class Weighted Loss on Class Distribution
    for next epoch's Unsupervised Data
    - Keep Weight Base as 1 (ie +1)
    - Use the Sqrt/ cube root of Distribution for the weighting
    - Contains both total and minority method
    - Also contains inversion if needed
    '''

    try :
        # Note: distbLoss does not have "darp" as an option
        distb = distb_dict[distbLoss]  
        print("Distribution Used: ", distb)
        if (distbLoss == "pseudo") and darp and (epoch > args.warm) :
            distb = distb_dict["darp"]
            print("DARPed: ", distb)
    except :
        weightLoss = None
        print("No Distribution was declared")

    try :   
        if (weightLoss["invert?"]) :
            distb = torch.flip(distb, dims=[0]) # Reverse the order of the Class's Weights
            # distb = invertDistribution(distb) # Beta Mode
        if (weightLoss["type"] == "total") :
            # Based on Proportion to Sum of all Dataset
            class_weight = distb / torch.sum(distb) * weightLoss["const"] + 1 
        elif (weightLoss["type"] == "minority") :
            lowest_ref = torch.min(distb)
            if (lowest_ref < 1) :
                lowest_ref = 1
            # Based on Proportion to Lowest Minority Class (Smallest = 1)
            class_weight = (distb / lowest_ref) ** (1/weightLoss["const"])
        elif (weightLoss["type"] == "intercept") :
            class_weight = distb / torch.sum(distb) + weightLoss["const"]
        elif (weightLoss["type"] == "log") :
            # Based on log Proportion to log Sum of all Dataset
            class_weight = torch.log(weightLoss["const"] * distb) \
                / torch.sum( torch.log(distb) )
        elif (weightLoss["type"] == "effective") :
            # Based on Effective Number of Samples
            N = torch.sum(distb) / len(distb) # Originally N is a hyperparameter
            beta = (N - 1) / N
            # beta = weightLoss["const"]
            print("beta used = ", beta)

            # Note: Having the power = 0 yields inf (1/0)
            ones = torch.ones(distb.shape)
            if use_cuda :
                ones = ones.cuda()
            distb = torch.where(distb == 0, ones, distb) 
            
            class_weight = (1 - beta) / (1 - (beta ** distb) )

            # class_weight = torch.sqrt(class_weight) # sqrt
        elif (weightLoss["type"] == "power") :
            # Based on Inversed-Powered Proportion of Total Distribution
            class_weight = (torch.sum(distb) / distb ) ** weightLoss["const"] 
            # Note: Proposed Alpha const: 0.7 

        if (weightLoss["normalize"] != None) :
            class_weight = weightLoss["normalize"] * len(class_weight ) \
                    * class_weight / torch.sum(class_weight) # torch.norm(class_weight, p=1)
    except :
        num_class = len(distb_dict["gt"]) # Using Ground Truth Class for Reference
        class_weight = torch.ones(num_class)
        if use_cuda :
            class_weight = class_weight.cuda()
        print("Using Uniform Weights ...")

    return class_weight