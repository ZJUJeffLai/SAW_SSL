import time
import torch

from utils import Bar, AverageMeter, accuracy
# from utils import Logger, mkdir_p, savefig

# # Augmentations for References
# from torchvision.transforms import transforms
# from RandAugment import RandAugment
# from RandAugment.augmentations import CutoutDefault
# 
# transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(cifar10_mean, cifar10_std)
#     ])

# transform_strong = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(cifar10_mean, cifar10_std)
# ])
# transform_strong.transforms.insert(0, RandAugment(3, 4))
# transform_strong.transforms.append(CutoutDefault(16))

# transform_val = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(cifar10_mean, cifar10_std)
# ])

def validate(valloader, num_class, model, criterion, use_cuda, mode):
    '''
    Evaluates your Model based on valloader dataset

    Inputs:
    valloader   : DataLoader Object containing dataset for Validation
    num_class   : (int) Number of Classes (for Classification)
    model       : Model to be tested
    criterion   : Loss Criterion for Model Evaluation
    use_cuda    : (bool) True if using CUDA, False if using CPU
    mode        : (str) To indicate which dataset is being evaluated

    Outputs:
        -  Average Loss
        -  Left Most Class Accuracy in Dataset (1st entry class)
        -  Average Accuracy (Numpy Obj)
        -  Geometric Mean
    '''

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    classwise_correct = torch.zeros(num_class)
    classwise_num = torch.zeros(num_class)
    section_acc = torch.zeros(3)

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)
            targets = targets.long() # For Old PyTorch

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # classwise prediction
            pred_label = outputs.max(1)[1]
            pred_mask = (targets == pred_label).float()
            for i in range(num_class):
                class_mask = (targets == i).float()

                classwise_correct[i] += (class_mask * pred_mask).sum()
                classwise_num[i] += class_mask.sum()

             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                          'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()

    # Major, Neutral, Minor
    section_num = int(num_class / 3)
    classwise_acc = (classwise_correct / classwise_num)
    section_acc[0] = classwise_acc[:section_num].mean()
    section_acc[2] = classwise_acc[-1 * section_num:].mean()
    section_acc[1] = classwise_acc[section_num:-1 * section_num].mean()
    GM = 1
    for i in range(num_class):
        if classwise_acc[i] == 0:
            # To prevent the N/A values, we set the minimum value as 0.001
            GM *= (1/(100 * num_class)) ** (1/num_class)
        else:
            GM *= (classwise_acc[i]) ** (1/num_class)

    return (losses.avg, top1.avg, section_acc.numpy(), GM)

def dojoTest(dataLoaders, num_class, ema_model, criterion, use_cuda) :
    '''
    For Robust Testing (like in Dojo Trainings)

    Inputs:
    dataLoaders : Dictionary containing DataLoaders for Dojo Training
        - Note: dojoTest() uses "Imbalanced", "Reversed", "Weak", Strong"
    num_class   : (int) Number of Classes (for Classification)
    model       : Model to be tested
    criterion   : Loss Criterion for Model Evaluation
    use_cuda    : (bool) True if using CUDA, False if using CPU

    Output:
    dojoStats   : list containing dojo-certified Statistics
    '''
    imb_test_loss, imb_test_acc, imb_test_cls, \
            imb_test_gm = validate(dataLoaders["Imbalanced"], num_class, ema_model, criterion, use_cuda, mode='Imbalanced Test Stats ')
    rev_test_loss, rev_test_acc, rev_test_cls, \
        rev_test_gm = validate(dataLoaders["Reversed"], num_class, ema_model, criterion, use_cuda, mode='Reversed Imbalanced Test Stats ')
    weak_test_loss, weak_test_acc, weak_test_cls, \
        weak_test_gm = validate(dataLoaders["Weak"], num_class, ema_model, criterion, use_cuda, mode='Weakly Augmented Balanced Test Stats ')
    strong_test_loss, strong_test_acc, strong_test_cls, \
        strong_test_gm = validate(dataLoaders["Strong"], num_class, ema_model, criterion, use_cuda, mode='Strongly Augmented Balanced Test Stats ')

    dojoStats = [imb_test_loss, imb_test_acc, imb_test_gm, \
        rev_test_loss, rev_test_acc, rev_test_gm, \
            weak_test_loss, weak_test_acc, weak_test_gm, \
                strong_test_loss, strong_test_acc, strong_test_gm]

    return dojoStats

# if __name__ == '__main__':
    # dojoTest()