#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:39:54 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard imports:
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import eye
from scipy.sparse import lil_matrix

# Chebpy imports:
from .sptoeplitz import sptoeplitz
from .trigpts import trigpts
from .vals2coeffs import vals2coeffs

def multmat(n, f, dom=[-1, 1]):
    """Return the n x n multiplication by f matrix in Fourier space."""
    # Get the Fourier coefficients:
    x = trigpts(4*n, dom)
    F = vals2coeffs(f(x))
    F = np.concatenate((F, np.array([F[0]/2])), axis=0)
    digits = int(15 - np.floor(np.log10(np.max(np.abs(F)))))
    F = np.round(F, digits)
    F[0] = 1/2*F[0]
        
    # Projection matrices:
    P = eye(n+1, n)
    P = lil_matrix(P)
    P[0, 0] = 1/2
    P[-1, 0] = 1/2
    col = np.zeros(n)
    row = np.zeros(2*n + 1)
    row[int(n/2)] = 1
    Q = sptoeplitz(col, row)
    Q = lil_matrix(Q)
    Q[0, n+2] = 1
    
    # Multiplication matrix:
    col = F[2*n:]
    row = np.flipud(F[:2*n+1])
    M = csr_matrix(sptoeplitz(col, row))
    
    # Truncate and project:
    M = Q @ M[:, int(n/2):3*int(n/2)+1] @ P
    
    return M