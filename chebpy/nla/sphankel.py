#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:49:44 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np
from scipy.sparse import triu

# Chebpy imports:
from .sptoeplitz import sptoeplitz

def sphankel(col):
    """Return a sparse Hankel matrix."""
    col = np.flipud(col)
    H = triu(sptoeplitz(col, col), format='csr')
    H = np.flip(H, axis=1)
    return H