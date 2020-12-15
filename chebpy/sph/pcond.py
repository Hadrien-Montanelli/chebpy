#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:28:26 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard imports:
import numpy as np

def pcond(F):
    """Test the pole condition for the DFS coefficients F."""
    n = len(F)
    sum1 = np.zeros(n)
    sum2 = np.zeros(n)
    for k in range(n):
        if (k == n/2):
            continue
        else:
            sum1[k] = abs(np.sum(F[:, k]))
            sum2[k] = abs(np.sum(F[:, k] * (-1)**np.arange(n)))
    P = max(np.max(sum1), np.max(sum2))
    return P