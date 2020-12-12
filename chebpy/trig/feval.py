#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:52:50 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
from math import pi
import numpy as np

def feval(F, x):
    """Evaluate Fourier series with coefficients F at x."""
    n = len(F)
    m = len(x)
    y = np.zeros(m)
    for k in range(m):
        if (n % 2 == 1):
            tmp = F * np.exp(1j * np.arange(-(n-1)/2, (n-1)/2+1) * pi * x[k])
            tmp = np.real(tmp)
            y[k] = np.sum(tmp)
        else:
            tmp = F * np.exp(1j * np.arange(-n/2, n/2)* pi * x[k])
            tmp = np.real(tmp)
            y[k] = np.sum(tmp)
    return y