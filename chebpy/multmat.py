#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:39:54 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard imports:
import numpy as np
from scipy.linalg import hankel
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix

# Chebpy imports:
from .chebpts import chebpts
from .vals2coeffs import vals2coeffs

def multmat(n, f):
    """Return the n x n multiplication by f matrix."""
    # Get the Chebyshev coefficients and scale first term:
    x = chebpts(n)
    coeffs = vals2coeffs(f(x))
    coeffs[0] = 2*coeffs[0]
    
    # Toeplitz part:
    T = toeplitz(coeffs)
    
    # Hankel part:
    H = hankel(coeffs[1:])
    H = np.concatenate((H, np.zeros([n-1, 1])), axis=1)
    H = np.concatenate((np.zeros([1, n]), H), axis=0)
    
    # Assemble M:
    M = csr_matrix(np.round(1/2*(T + H), 15))
    
    return M
