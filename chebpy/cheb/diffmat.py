#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:29:07 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np
from scipy.sparse import eye
from scipy.sparse import spdiags

def diffmat(n, m, dom=[-1, 1]):
    """Return the n x n mth-order differentiation matrix in Chebyshev space."""
    if (m>0):
        diag = [j for j in range(n)]
        D = spdiags(diag, 1, n, n)
        for s in range(1, m):
            diag = 2*s*np.ones(n)
            D = spdiags(diag, 1, n, n)*D
        D = (2/(dom[1] - dom[0]))**m*D
    else:
        D = eye(n)
    return D