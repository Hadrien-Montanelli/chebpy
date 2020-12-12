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
        y[k] = np.sum(np.real(F * np.exp(1j*np.arange(-n/2, n/2)*pi*x[k])))
    return y