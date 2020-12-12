#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:28:26 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard imports:
from math import pi
import numpy as np

def mean(F):
    """Return the mean of the DFS series with coefficients F."""
    n = len(F)
    nn = np.arange(-n/2, n/2)
    nn[int(n/2)-1] = 0
    nn[int(n/2)+1] = 0
    en = 2*pi*(1 + np.exp(1j*pi*nn))/(1 - nn**2)
    mu = np.real(en @ F[:, int(n/2)])
    return mu