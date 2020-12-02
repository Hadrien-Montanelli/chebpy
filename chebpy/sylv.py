#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 17:38:32 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library import:
import numpy as np
from scipy.linalg import solve
from scipy.linalg import schur

def sylv(A, B, C):
    
    # Compute the Schur decomposotions:
    R, U = schur(A)
    S, V = schur(B)

    # Assemble D:
    D = U.T @ C @ V
    
    # Solve RY - YS = D for Y:
    n = len(A)
    Y = np.zeros([n, n])
    I = np.eye(n)
    for j in range(n):
        LHS = R - S[j, j]*I
        RHS = D[:, j]
        for k in range(j):
            RHS += S[k, j]*Y[:, k]
        Y[:, j] = solve(LHS, RHS)
    
    # Assemble solution:
    X =  U @ Y @ V.T
    
    return X