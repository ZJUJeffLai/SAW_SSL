#!/bin/bash

#########################################
# SECTION 1: Variables                  #
#########################################

# Variables
FOLDERNAME="result28_beta_0.999_100_100"
DATASET="cifar10"       # "cifar10" for CIFAR-10, "cifar100" for CIFAR100
BASEDIR="experiments"
FOLDERDIR="${BASEDIR}/${DATASET}/FixMatch/${FOLDERNAME}"
SETTINGTXT="${FOLDERDIR}/details.txt"
PROGRESSTXT="${FOLDERDIR}/progress.txt"

# Experiment Settings
EPOCHS=500
START=0
BATCH=64
LEARNING_RATE=0.002
W_DECAY=0   # For Adam/SGD Optimizer

# Method Options (Class Dataset & Model Settings)
NUM_MAX=1500    # 1500 for CIFAR-10, 150 for CIFAR-100
RATIO=2.0
IMB_L=100
IMB_U=100
VAL_ITER=500
NUM_VAL=10

# Hyperparameters for FixMatch
TAU=0.95
EMA_DECAY=0.999
LAMBDA_U=1  # If normalized, sum(weights) = lambda_u * num_class

# Hyperparameters for FixMatch
WARM=200
ALPHA=2.0
DARP="" # "--darp" to use DARP, "" to NOT use DARP
EST="" # "--est" to use Estimated Distribution for Unlabeled Dataset
        # "" to NOT use Estimated Distribution for Unlabeled Dataset
ITER_T=10
NUM_ITER=10

# Settings Used for Weighted Loss based on Class Distribution
W_L="--effective"   # "--total" to use Total (Class/Total) Weighting Scheme (W_L = [1, CONST+1])
                # "--minority" to use Minority (Class/Minority) Weighting Scheme (W_L = [1, INF])
                # "--effective" to use Effective Number Weighting Scheme 
                #           Note: Naturally Penalizes Minority than Majority
                # "--power" to use Power (Class/Total)^alpha Weighting Scheme (CONST=alpha hyperparameter)
                #           Note: Naturally Penalizes Minority than Majority
                # "" to use Uniform (Ones) Class Weighting Scheme

CONST=0.999         # Hyperparameter for W_L Formula (Comment if not using)

# Distribution used for Loss in Labeled Data
DISTBL="gt_l"   # "uniform": To use Uniform Weights (All Ones, similar to default FixMatch)
                # "gt_l"   : To use Ground Truth (Labeled) Class Distribution for Weighting Scheme (Used in Training)

# Distribution used for Loss in Unlabeled Data
DISTBU="gt" # "uniform": To use Uniform Weights (All Ones, similar to default FixMatch)  
                # "pseudo" : To use Pseudo-Label Distribution for Weighting Scheme
                # "weak"   : To use Weakly Augmented Output Distribution for Weighting Scheme
                # "strong" : To use Strongly Augmented Output Distribution for Weighting Scheme
                # "gt"     : To use Ground Truth (All) Class Distribution for Weighting Scheme (For Justification)
                # "gt_l"   : To use Ground Truth (Labeled) Class Distribution for Weighting Scheme (For Justification)
                # "gt_u"   : To use Ground Truth (UnLabeled) Class Distribution for Weighting Scheme (For Justification)

INV=""  # "--invert" : To flip class weights on Loss (Penalize Minority more than Majority)
                # "" : To leave class weights to original on Loss (Penalize Majority more than Minority)
                # Note: Please check the Weighting Scheme to see its natural effects

NORM="--normalize" # "--normalize" : To normalize class weights on Loss according
                   #                 to number of classes after Weighting Function Calculation
                   # ""            : To leave class weights according to Weighting Function Calculation

NORMVAL=1       # Hyperparameter for Normalization
                # such that sum(weights) = NORMVAL * num_class

# For More Info: Execute "python3 train_fix.py --help"

#########################################
# SECTION 2: Print Receipt              #
#########################################

mkdir ${BASEDIR}/${DATASET}
mkdir ${BASEDIR}/${DATASET}/FixMatch
mkdir $FOLDERDIR 
mkdir "${FOLDERDIR}/checkpoints" 

# Print Settings for Reference
echo -e \
"Settings Used for $FOLDERNAME @ $FOLDERDIR : \n \
\n \
Dataset       = $DATASET \n \
\n \
Total Epochs  = $EPOCHS \n \
Start Epoch   = $START \n \
Batch Size    = $BATCH # (for each Supervised & Unsupervised) \n \
Learning Rate = $LEARNING_RATE # Learning Rate \n \
Weight Decay  = $W_DECAY # Used for Optimizer (Adam, SGD, etc.) \n \
\n \
# Method Options (Class Dataset & Model Settings) \n \
Number of Samples in Maximal Class = $NUM_MAX \n \
Labeled vs Unlabeled Data Ratio    = $RATIO \n \
Imbalance Ratio for Labeled Data   = $IMB_L \n \
Imbalance Ratio for Unlabeled Data = $IMB_U \n \
Frequency of Evaluation            = $VAL_ITER \n \
Number of Validation Data          = $NUM_VAL \n \
\n \
# Hyperparameters for FixMatch \n \
Minimal Confidence for Pseudo-Label (tau) = $TAU \n \
EMA Decay Hyperparameter                  = $EMA_DECAY \n \
Weight for Unsupervised Loss (Lambda_u)   = $LAMBDA_U \n \
\n \
# Hyperparameters for DARP \n \
Warmup Epoch                              = $WARM \n \
Hyperparameter for removing Noisy Entries = $ALPHA \n " > $SETTINGTXT

if [ "$DARP" = "" ] ; then
    echo -e "Using DARP = False" >> $SETTINGTXT
elif [ "$DARP" = "--darp" ] ; then
    echo -e "Using DARP = True"  >> $SETTINGTXT
else
    echo -e "Error in DARP Setting, Please Correct it. Exiting..." \
    | tee -a $SETTINGTXT
    exit 1
fi

if [ "$EST" = "" ] ; then
    echo -e "Using Estimated Distribution for Unlabeled Dataset = False" >> $SETTINGTXT
elif [ "$EST" = "--est" ] ; then
    echo -e "Using Estimated Distribution for Unlabeled Dataset = True"  >> $SETTINGTXT
else
    echo -e "Error in EST Setting, Please Correct it. Exiting..." \
    | tee -a $SETTINGTXT
    exit 1
fi

echo -e \
"Number of iteration (T) for DARP          = $ITER_T \n \
Scheduling for updating Pseudo-Labels      = $NUM_ITER \n \
\n \
# Settings Used for Weighted Loss based on Class Distribution \n \
CONST = $CONST \n" >> $SETTINGTXT

if [ "$INV" == "--invert" ] ; then
    echo -e "INV = True \n" >> $SETTINGTXT
elif [ "$INV" == "" ] ; then
    echo -e "INV = False \n" >> $SETTINGTXT
else
    echo -e "Error in INV Setting, Please Correct it. Exiting..." \
    | tee -a $SETTINGTXT
    exit 1
fi

if [ "$NORM" == "--normalize" ] ; then
    echo -e "Normalize = ${NORMVAL} \n" >> $SETTINGTXT
elif [ "$NORM" == "" ] ; then
    echo -e "Normalize = None \n" >> $SETTINGTXT
else
    echo -e "Error in NORM Setting, Please Correct it. Exiting..." \
    | tee -a $SETTINGTXT
    exit 1
fi

if [ "$W_L" = "--total" ] ; then
    echo -e "Weight Loss Formula Used (Based on Sum Distribution): \n \
    class_weight = distb / torch.sum(distb) * (x = ${CONST}) + 1 \n \
    Note: class_weight = [1, $(expr ${CONST} + 1)] where Hyperparameter x=${CONST} \n" >> $SETTINGTXT
elif [ "$W_L" = "--minority" ] ; then
    echo -e "Weight Loss Formula Used (Based on Minority Distribution): \n \
    class_weight = (distb / lowest_ref) ** [ 1 / (x = ${CONST}) ] \n \
    Note: class_weight = [1, inf] where Hyperparameter x=${CONST} \n"  >> $SETTINGTXT
elif [ "$W_L" = "--effective" ] ; then
    echo -e "Weight Loss Formula Used (Based on Effective Number of Samples): \n \
    class_weight = class_weight = (1 - beta) / (1 - (beta ** distb) \n \
        where beta = ${CONST} \n \
    Note: class_weight = [0, 1] with an automatic Hyperparameter \n"  >> $SETTINGTXT
elif [ "$W_L" = "--power" ] ; then
    echo -e "Weight Loss Formula Used (Based on Inversed, Powered-Sum Distribution): \n \
    class_weight = (torch.sum(distb) / distb ) ** (alpha = ${CONST})  \n \
    Note: class_weight = [1, inf] where Hyperparameter alpha=${CONST} \n"  >> $SETTINGTXT
elif [ "$W_L" = "" ] ; then
    echo -e "Using Equal Weighting (torch.Ones(num_class))" >> $SETTINGTXT
else
    echo -e "Error in W_L Setting, Please Correct it. Exiting..." \
    | tee -a $SETTINGTXT
    exit 1
fi

if [ "$INV" == "--invert" ] ; then
    echo -e "2nd Note: Distribution Weighting (distb) was flipped \n" >> $SETTINGTXT
elif [ "$INV" == "" ] ; then
    echo -e "2nd Note: Distribution Weighting (distb) was NOT flipped \n" >> $SETTINGTXT
else
    echo -e "Error in INV Setting, Please Correct it. Exiting..." \
    | tee -a $SETTINGTXT
    exit 1
fi

if [ "$NORM" == "--normalize" ] ; then
    echo -e "3rd Note: Class Weights were normalized according to Number of Classes \
        \n    ie. class_distb = len(distb) * class_distb / sum(class_distb) \
        \n    This results: sum(class_distb) = num_class * (norm_val = $NORMVAL) \
        \n                                     = $(expr ${NORMVAL} \* 10) \n" >> $SETTINGTXT
elif [ "$NORM" == "" ] ; then
    echo -e "3rd Note: Weighting Scheme was NOT normalized (left as original) \n" >> $SETTINGTXT
else
    echo -e "Error in NORM Setting, Please Correct it. Exiting..." \
    | tee -a $SETTINGTXT
    exit 1
fi

echo -e "\n" >> $SETTINGTXT

if [ "$DISTBU" = "pseudo" ] ; then
    echo -e "Class Distribution used for Unlabeled Loss  = Psuedo-Label (p) \n" >> $SETTINGTXT
elif [ "$DISTBU" = "weak" ] ; then
    echo -e "Class Distribution used for Unlabeled Loss  = Weakly Augmented Model Prediction (p_hat) \n"  >> $SETTINGTXT
elif [ "$DISTBU" = "strong" ] ; then
    echo -e "Class Distribution used for Unlabeled Loss  = Strongly Augmented Model Prediction (q) \n"  >> $SETTINGTXT
elif [ "$DISTBU" = "gt" ] ; then
    echo -e "Class Distribution used for Unlabeled Loss  = Ground Truth Distribution (Labeled + Unlabeled) \n"  >> $SETTINGTXT
elif [ "$DISTBU" = "gt_l" ] ; then
    echo -e "Class Distribution used for Unlabeled Loss  = Ground Truth Distribution (Labeled) \n"  >> $SETTINGTXT
elif [ "$DISTBU" = "gt_u" ] ; then
    echo -e "Class Distribution used for Unlabeled Loss  = Ground Truth Distribution (Unlabeled) \n    "  >> $SETTINGTXT
elif [ "$DISTBU" = "uniform" ] || [ "$W_L" = "" ] ; then
    echo -e "No Class Distribution was used for Unlabeled Loss (Formula Does NOT Apply)" >> $SETTINGTXT
else
    echo -e "Error in DISTBU Setting, Please Correct it. Exiting..." \
    | tee -a $SETTINGTXT
    exit 1
fi

if [ "$DISTBL" = "pseudo" ] ; then
    echo -e "Class Distribution used for Labeled Loss  = Psuedo-Label (p) \n" >> $SETTINGTXT
elif [ "$DISTBL" = "weak" ] ; then
    echo -e "Class Distribution used for Labeled Loss  = Weakly Augmented Model Prediction (p_hat) \n"  >> $SETTINGTXT
elif [ "$DISTBL" = "strong" ] ; then
    echo -e "Class Distribution used for Labeled Loss  = Strongly Augmented Model Prediction (q) \n"  >> $SETTINGTXT
elif [ "$DISTBL" = "gt" ] ; then
    echo -e "Class Distribution used for Labeled Loss  = Ground Truth Distribution (Labeled + Unlabeled) \n"  >> $SETTINGTXT
elif [ "$DISTBL" = "gt_l" ] ; then
    echo -e "Class Distribution used for Labeled Loss  = Ground Truth Distribution (Labeled) \n"  >> $SETTINGTXT
elif [ "$DISTBL" = "gt_u" ] ; then
    echo -e "Class Distribution used for Labeled Loss  = Ground Truth Distribution (Unlabeled) \n    "  >> $SETTINGTXT
elif [ "$DISTBL" = "uniform" ] || [ "$W_L" = "" ] ; then
    echo -e "No Class Distribution was used for Labeled Loss (Formula Does NOT Apply)" >> $SETTINGTXT
else
    echo -e "Error in DISTBL Setting, Please Correct it. Exiting..." \
    | tee -a $SETTINGTXT
    exit 1
fi

#########################################
# SECTION 3: Execute Experiments        #
#########################################

# Execute Experiment
python3 train_fix_${DATASET}.py \
--epochs $EPOCHS \
--start-epoch $START \
--batch-size $BATCH \
--lr $LEARNING_RATE \
--wd $W_DECAY \
\
--num_max $NUM_MAX \
--ratio $RATIO \
--imb_ratio_l $IMB_L \
--imb_ratio_u $IMB_U \
--val-iteration $VAL_ITER \
--num_val $NUM_VAL \
\
--tau $TAU \
--ema-decay $EMA_DECAY \
--lambda_u $LAMBDA_U \
\
--warm $WARM \
--alpha $ALPHA \
$DARP \
$EST \
--iter_T $ITER_T \
--num_iter $NUM_ITER \
\
$W_L $CONST \
--distbl $DISTBL \
--distbu $DISTBU \
$INV \
$NORM $NORMVAL \
\
--out $FOLDERDIR \
\
| tee $PROGRESSTXT

# Note: Use tee -a to append text

#########################################
# SECTION 4: Analyze Experiments        #
#########################################

python3 analysis.py \
--out $FOLDERDIR

# End of Experiment