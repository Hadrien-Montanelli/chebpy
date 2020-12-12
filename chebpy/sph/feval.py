#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:52:50 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
from math import pi
import numpy as np

def feval(f, LAM, TT):
    """Evaluate doubled-up version of f on the sphere."""
    n = len(LAM)
    Y = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if (TT[i, 0] >= 0):
                Y[i, j] = f(LAM[0, j], TT[i, 0])
            else:
                if (LAM[0, j] <=0):
                    Y[i, j] = f(LAM[0, j] + pi, -TT[i, 0])
                else:
                    Y[i, j] = f(LAM[0, j] - pi, -TT[i, 0])
    return Y