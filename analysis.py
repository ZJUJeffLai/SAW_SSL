import argparse
import os

from utils import Logger
import numpy as np


parser = argparse.ArgumentParser(description='Analysis of Log Files')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

def confusion(model, loader, num_class):
    model.eval()

    num_classes = torch.zeros(num_class)
    confusion = torch.zeros(num_class, num_class)

    for batch_idx, (inputs, targets) in enumerate(loader):
        batch_size = inputs.size(0)
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs, _ = model(inputs)
        probs = torch.softmax(outputs.data, dim=1)

        # Update the confusion matrix
        for i in range(batch_size):
            confusion[:, targets[i]] += probs[i].cpu()
            num_classes[targets[i]] += 1

    return confusion

def main() :
    mainLog = Logger(os.path.join(args.out, 'log.txt'), resume=True)
    print("mainLog.names: ", mainLog.names)

    

    interested = ['Test Acc.']

    for name in interested : # mainLog.names :
        # print("name: ", name)
        # print("contains: ", type(mainLog.numbers[name]))
        npAcc = np.asarray([float(x) for x in mainLog.numbers[name]])
        # print("npAcc = ", npAcc)
        maxValue = np.amax(npAcc)
        maxIdx = np.argmax(npAcc)

        print(name, "'s max value = ", maxValue, \
            " at Index ", maxIdx)

    testLog = Logger(os.path.join(args.out, 'testLog.txt'), resume=True)
        # print("testLog.names: ", testLog.names)

    interested = ['Imbalanced Acc.', 'Imbalanced GM.', 'Reversed Acc.', 'Reversed GM.', \
         'Weak Acc.', 'Weak GM.', 'Strong Acc.', 'Strong GM.']
    for name in interested : # testLog.names :
        # print("name: ", name)
        # print("contains: ", type(mainLog.numbers[name]))
        npAcc = np.asarray([float(x) for x in testLog.numbers[name]])
        # print("npAcc = ", npAcc)
        maxValue = np.amax(npAcc)
        maxIdx = np.argmax(npAcc)

        print(name, "'s max value = ", maxValue, \
            " at Index ", maxIdx)

    # paths = {"1.5:1": ,
    #         "2:1" : ,
    #         "3:1" : , 
    #         }

if __name__ == '__main__':
    main()