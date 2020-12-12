#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:00:03 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np
from scipy.linalg import qz
from scipy.linalg import solve

def gensylv(A, B, C, D, E):
    """Solve the generalized Sylvester equation AXB^T + CXD^T = E."""
    # Compute the QZ decompositions:
    AA, CC, Q1, Z1 = qz(A, C) # Q1, Z1, A, C are m x m
    DD, BB, Q2, Z2 = qz(D, B) # Q2, Z2, A, C are n x n
    Q1 = Q1.T # Python's convention is not standard
    Q2 = Q2.T

    # Assemble F:
    F = Q1 @ E @ Q2.T # E, F are m x n
    
    # Assemble P and S:
    P = Q1 @ A @ Z1 # P is m x m 
    S = Q1 @ C @ Z1 # S is m x m 
    
    # Assemble R and T:
    R = Q2 @ B @ Z2 # R is n x n
    T = Q2 @ D @ Z2 # T is n x n
    
    # Solve PYR^T + SYT^T = F for Y:
    m = len(A)
    n = len(B)
    Y = np.zeros([m, n]) # Y is m x n
    k = n - 1
    while (k>=0):
        if (np.abs(T[k, k-1])<1e-14 or k==0):
            LHS = R[k, k]*P + T[k, k]*S
            RHS = F[:, k]
            for j in range(k+1, n):
                RHS -= (R[k, j]*P + T[k, j]*S) @ Y[:, j]
            Y[:, k] = solve(LHS, RHS)
            k -= 1
        else:   
            LHS = np.zeros([2*m, 2*m])
            LHS[:m, :m] = R[k-1, k-1]*P + T[k-1, k-1]*S
            LHS[:m, m:2*m] = R[k-1, k]*P + T[k-1, k]*S
            LHS[m:2*m, :m] = T[k, k-1]*S
            LHS[m:2*m, m:2*m] = R[k, k]*P + T[k, k]*S
            RHS = np.zeros(2*m)
            RHS[:m] = F[:, k-1]
            for j in range(k, n):
                RHS[:m] -= (R[k-1, j]*P + T[k-1, j]*S) @ Y[:, j]
            RHS[m:2*m] = F[:, k]
            for j in range(k+1, n):
                RHS[m:2*m] -= (R[k, j]*P + T[k, j]*S) @ Y[:, j]
            tmp = solve(LHS, RHS)
            Y[:, k-1] = tmp[:m]
            Y[:, k] = tmp[m:2*m]
            k -= 2
        
    # Assemble solution:
    X = Z1 @ Y @ Z2.T # X is m x n
    
    return X