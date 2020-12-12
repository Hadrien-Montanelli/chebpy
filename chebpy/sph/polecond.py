#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:28:26 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard imports:
import numpy as np

def polecond(F):
    """Return the pole condition for the DFS coefficients F."""
    n = len(F)
    rowsum1 = np.zeros(n)
    rowsum2 = np.zeros(n)
    for k in range(n):
        if (k == n/2):
            continue
        else:
            rowsum1[k] = np.abs(np.sum(F[:, k]))
            rowsum2[k] = np.abs(np.sum(F[:, k] * (-1)**(k-n/2)))
    P = max(np.max(rowsum1), np.max(rowsum2))
    return P