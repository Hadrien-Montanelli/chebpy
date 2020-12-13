#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:49:29 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard imports:
import numpy as np

def fpcond(F):
    """Enforce the pole condition for the DFS coefficients F."""
    
    # Get the dimension:
    n = len(F)

    # Negative wavenumbers:
    G = F[:, :int(n/2)]
    A = np.ones([2, n])
    A[1, :] = (-1)**np.arange(0, n)
    C = A.T @ (np.linalg.inv(A @ A.T) @ A @ G)
    F[:, :int(n/2)] = F[:, :int(n/2)] - C
    
    # Positive wavenumbers:
    G = F[:, int(n/2)+1:]
    A = np.ones([2, n])
    A[1, :] = (-1)**np.arange(1, n+1)
    C = A.T @ (np.linalg.inv(A @ A.T) @ A @ G)
    F[:, int(n/2)+1:] = F[:, int(n/2)+1:] - C
    
    return F