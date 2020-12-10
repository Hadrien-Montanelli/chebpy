#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:47:44 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np
from scipy.sparse import spdiags

def spconvert(n, lam):
    """Return the n x n C^{(lam)} to C^{(lam+1)} conversion matrix."""
    if (lam == 0):
        diags = .5*np.ones([2, n])
        diags[0, 0] = 1
        diags[1, :] = -diags[1, :]
        S = spdiags(diags, [0, 2], n, n)
    else:
        diags = np.zeros([2, n])
        diags[0,:] = lam/(lam + np.arange(n))
        diags[1,:] = lam/(lam + np.arange(n))
        diags[0, 0] = 1
        diags[1, :] = -diags[1, :]
        S = spdiags(diags, [0, 2], n, n)
    return S