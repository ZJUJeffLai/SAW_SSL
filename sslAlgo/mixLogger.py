import os

from utils import Logger

def createLogger(path, num_class=10, title="") :
    print("Creating Logger File at: ", path+'/log.txt')

    loggerDict = {}
    loggerDict["logger"] = Logger(os.path.join(path, 'log.txt'), title=title)
    loggerDict["logger"].set_names(['Train Loss', 'Train Loss X', 'Train Loss U',  \
            'Train Acc X', 'Train GM X', \
            'Test Loss', 'Test Acc.', 'Test GM.'])
    loggerDict["dojo"] = Logger(os.path.join(path, 'testLog.txt'), title=title)
    loggerDict["dojo"].set_names(['Imbalanced Loss', 'Imbalanced Acc.', 'Imbalanced GM.',  \
            'Reversed Loss', 'Reversed Acc.', 'Reversed GM.', \
                'Weak Loss', 'Weak Acc.', 'Weak GM.', \
                    'Strong Loss', 'Strong Acc.', 'Strong GM.' ])

    # loggerDict["pseudo"] = Logger(os.path.join(path, 'pseudo_distb.txt'), title=title)
    # loggerDict["pseudo"].set_names(['0-Major', '1', '2', '3', '4', '5', \
    #     '6', '7', '8', '9-Minor'])
    # loggerDict["darp"] = Logger(os.path.join(path, 'darp_distb.txt'), title=title)
    # loggerDict["darp"].set_names(['0-Major', '1', '2', '3', '4', '5', \
    #     '6', '7', '8', '9-Minor'])
    # loggerDict["weak"] = Logger(os.path.join(path, 'weak_distb.txt'), title=title)
    # loggerDict["weak"].set_names(['0-Major', '1', '2', '3', '4', '5', \
    #     '6', '7', '8', '9-Minor'])
    # loggerDict["strong"] = Logger(os.path.join(path, 'strong_distb.txt'), title=title)
    # loggerDict["strong"].set_names(['0-Major', '1', '2', '3', '4', '5', \
    #     '6', '7', '8', '9-Minor'])

    loggerDict["pseudo"] = Logger(os.path.join(path, 'pseudo_distb.txt'), title=title)
    loggerDict["pseudo"].set_names([str(c) for c in range(num_class)])
    loggerDict["darp"] = Logger(os.path.join(path, 'darp_distb.txt'), title=title)
    loggerDict["darp"].set_names([str(c) for c in range(num_class)])
    # loggerDict["weak"] = Logger(os.path.join(path, 'weak_distb.txt'), title=title)
    # loggerDict["weak"].set_names([str(c) for c in range(num_class)])
    # loggerDict["strong"] = Logger(os.path.join(path, 'strong_distb.txt'), title=title)
    # loggerDict["strong"].set_names([str(c) for c in range(num_class)])

    return loggerDict

def loadLogger(path, title="") :
    loggerDict = {}
    loggerDict["logger"] = Logger(os.path.join(path, 'log.txt'), title=title, resume=True)
    loggerDict["dojo"] = Logger(os.path.join(path, 'testLog.txt'), title=title, resume=True)
    loggerDict["pseudo"] = Logger(os.path.join(path, 'pseudo_distb.txt'), \
        title=title+" Pseudo-Label (p) (no DARP) Distribution", resume=True)
    loggerDict["darp"] = Logger(os.path.join(path, 'darp_distb.txt'), \
        title=title+" Pseudo-Label (p) (w/ DARP) Distribution", resume=True)
    # loggerDict["weak"] = Logger(os.path.join(path, 'weak_distb.txt'), \
    #     title=title+" Weakly Augmented Output (p_hat) Distribution", resume=True)
    # loggerDict["strong"] = Logger(os.path.join(path, 'strong_distb.txt'), \
    #     title=title+" Strongly Augmented Prediction (q) Distribution", resume=True)

    return loggerDict

def appendLogger(stats, dojoStats, distb_dict, loggerDict, printer=False) :
    # torch -> list
    # pseudo_distb_u = distb_dict["pseudo"].cpu().detach().tolist()
    # pseudo_distb_u = [int(p) for p in pseudo_distb_u]
    # darp_distb_u = distb_dict["darp"].detach().tolist()
    # darp_distb_u = [int(p) for p in darp_distb_u] 
    # weak_distb_u = distb_dict["weak"].cpu().detach().tolist()
    # weak_distb_u = [int(p) for p in weak_distb_u]  
    # strong_distb_u = distb_dict["strong"].cpu().detach().tolist()
    # strong_distb_u = [int(p) for p in strong_distb_u]
    
    # if printer :
    #     print("Distribution (Pseudo-Label) = ", pseudo_distb_u)
    #     print("Distribution (Pseudo-Label, DARPed) = ", darp_distb_u)
    #     print("Weakly Augmented Distribution (Unsupervised) = ", weak_distb_u)
    #     print("Strongly Augmented Distribution (Unsupervised) = ", strong_distb_u)

    for distbType in distb_dict :
        if distbType.startswith("gt") :
            continue

        distb_u = distb_dict[distbType].cpu().detach().tolist()
        distb_u = [int(p) for p in distb_u]

        if printer :
            print("Distribution for ", distbType, " = ", distb_u)

        loggerDict[distbType].append(distb_u)

    loggerDict["logger"].append(stats)
    loggerDict["dojo"].append(dojoStats)
    # loggerDict["pseudo"].append(pseudo_distb_u)
    # loggerDict["darp"].append(darp_distb_u)
    # loggerDict["weak"].append(weak_distb_u)
    # loggerDict["strong"].append(strong_distb_u)

    return loggerDict

def closeLogger(loggerDict) :
    for key in loggerDict :
        loggerDict[logger].close()
    
    # loggerDict["logger"].close()
    # loggerDict["pseudo"].close()
    # loggerDict["darp"].close()

    # Fix Match
    # loggerDict["weak"].close()
    # loggerDict["strong"].close()

    return loggerDict