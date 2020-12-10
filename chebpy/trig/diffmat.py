#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:29:07 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
from math import pi
import numpy as np
from scipy.sparse import eye
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags

def diffmat(n, m, dom=[-1, 1]):
    """Return the n x n mth-order differentiation matrix in Fourier space."""
    if (m > 0):
        P = np.eye(n+1, n)
        P[0, 0] = 1/2
        P[-1, 0] = 1/2
        P = csr_matrix(P)
        Q = np.eye(n, n + 1)
        Q[0, -1] = 1
        Q = csr_matrix(Q)
        D = (1j*spdiags(np.arange(-n/2,n/2+1), 0, n+1, n+1))**m
        D = Q @ D @ P
        D = (2*pi/(dom[1] - dom[0]))**m*D
    else:
        D = eye(n)
    return D