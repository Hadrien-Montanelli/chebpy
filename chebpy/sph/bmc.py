#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:28:26 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard imports:
import numpy as np

def bmc(F):
    """Test the BMC-I symmetry for the DFS coefficients F."""
    n = len(F)
    S = 0
    for k in range(1, n):
        for j in range(1, int(n/2)):
            S = max(S, abs(F[j, k] - F[n-j, k] * (-1)**(k-n/2)))
    for j in range(1, n):
        for k in range(1, int(n/2)):
            S = max(S, abs(F[j, k] - np.conj(F[j, n-k]) * (-1)**(k-n/2)))
    return S