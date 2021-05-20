import numpy as np

import argparse
import os
import sys
import json 
import logging

from conf import * # import config parameters

from scripts import Dimer_Dynamics


if __name__ == "__main__":

    print('job start')

    parser = argparse.ArgumentParser()
    parser.add_argument("--h", help = "h",type=float)
    parser.add_argument("--q", help = "q",type=float)
    args = parser.parse_args()
    h = round(float(1),2) if (not args.h) and (args.h != 0) else round(args.h,2)
    q = round(float(1),2) if (not args.q) and (args.q != 0) else round(args.q,2)

    # n_chains = 10
    # n_samples = int(3e6)
    # n_jobs = -1

    # length = [4,4]
    # alpha = 2
    # t_list = np.linspace(0, 30, 101)

    V = h * q


    Dimer_Dynamics(h, V, length,  alpha ,t_list, n_jobs = n_jobs, n_chains = n_chains, n_samples = n_samples, NUM=NUM, n_max = n_max)

