#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:39:54 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard imports:
import numpy as np
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix

# Chebpy imports:
from .trigpts import trigpts
from .vals2coeffs import vals2coeffs

import time

def multmat(n, f, dom=[-1, 1]):
    """Return the n x n multiplication by f matrix in Fourier space."""
    # Get the Fourier coefficients:
    start = time.time()
    x = trigpts(4*n, dom)
    F = vals2coeffs(f(x))
    F = np.concatenate((F, np.array([F[0]/2])), axis=0)
    F[0] = 1/2*F[0]
    end = time.time()
    print(f'Time (setup):  {end-start:.5f}s')
    
    # Projection matrices:
    start = time.time()
    P = np.eye(n+1, n)
    P[0, 0] = 1/2
    P[-1, 0] = 1/2
    P = csr_matrix(P)
    Q = np.eye(n, n + 1)
    Q[0, -1] = 1
    Q = np.concatenate((np.zeros([n, int(n/2)]), Q), axis=1)
    Q = np.concatenate((Q, np.zeros([n, int(n/2)])), axis=1)
    Q = csr_matrix(Q)
    end = time.time()
    print(f'Time (setup):  {end-start:.5f}s')
    
    # Multiplication matrix:
    start = time.time()
    col = np.round(F[2*n:], 13)
    row = np.round(np.flipud(F[:2*n+1]), 13)
    M = csr_matrix(np.round(toeplitz(col, row), 13))
    end = time.time()
    print(f'Time (setup):  {end-start:.5f}s')
    
    # Truncate and project:
    start = time.time()
    M = Q @ M[:, int(n/2):3*int(n/2)+1] @ P
    end = time.time()
    print(f'Time (setup):  {end-start:.5f}s')
    
    return M