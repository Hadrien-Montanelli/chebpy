#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:49:44 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np
from scipy.sparse import spdiags

def sptoeplitz(col, row):
    """Return a sparse toeplitz matrix T = toeplitz(col, row)."""
    # Get the dimensions:
    m = len(col)
    n = len(row)
    
    # Get the non-zero column and row elements:
    idx_col = np.where(col != 0)[0]
    idx_row = np.where(row != 0)[0]
    idx_row = idx_row[idx_row != 0]
    idx = np.concatenate((-idx_col, idx_row), axis=0)
    
    # Construct diagonals for spdiags:
    diags_col = col[idx_col]
    diags_col = np.tile(diags_col, (max(m, n), 1))
    diags_row = row[idx_row]
    diags_row = np.tile(diags_row, (max(m, n), 1))
    diags = np.concatenate((diags_col, diags_row), axis=1)
    
    # Create a sparse matrix via spiags:
    T = spdiags(diags.T, idx, m, n)
    
    return T
