import numpy as np

import argparse
import os
import sys
import json 
import logging

from scripts import Dimer_Corr

from conf import * 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--h", help = "h",type=float)
    parser.add_argument("--q", help = "q",type=float)
    parser.add_argument("--n", help = "n",type=int)
    args = parser.parse_args()
    h = round(1.0,2) if (not args.h) and (args.h != 0) else round(args.h,2)
    q = round(1.0,2) if (not args.q) and (args.q != 0) else round(args.q,2)
    n = 1 if (not args.n) and (args.n != 0) else round(args.n,2)
    n = int(n)

    # n_samples = int(3e6)
    # a = 0

    # length = [4,4]
    # t_list = np.linspace(0, 30, 101)

    V = h * q



    Dimer_Corr(h, V, length, t_list, n_samples_corr, a, n)

