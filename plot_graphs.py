import argparse
import os

from utils import Logger, LoggerMonitor, savefig
import numpy as np

BASE_DIR = "/root/imbalancedSSL/experiments/"
LOG_TXT = "log.txt"

def main() :
    paths = {
    '10:1': BASE_DIR + "result22/" + LOG_TXT, 
    '5:1': BASE_DIR + "result19/" + LOG_TXT,
    '3:1': BASE_DIR + "result23/" + LOG_TXT,
    '2:1': BASE_DIR + "result25/" + LOG_TXT,
    '1.5:1': BASE_DIR + "result21/" + LOG_TXT,
    '1:2': BASE_DIR + "result26/" + LOG_TXT,
    '1:3': BASE_DIR + "result24/" + LOG_TXT,
    '1:5': BASE_DIR + "result20/" + LOG_TXT,
    '1:20': BASE_DIR + "result20/" + LOG_TXT,
    '1:40': BASE_DIR + "result21/" + LOG_TXT,
    }

    field = ['Test Acc.']

    monitor = LoggerMonitor(paths)
    monitor.plot(names=field)
    savefig('ratio_l_u_100.jpg', dpi=600)

if __name__ == '__main__':
    main()