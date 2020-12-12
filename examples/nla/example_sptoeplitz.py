#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:59:37 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
import numpy as np
from scipy.sparse import csr_matrix

# Chebpy imports:
from chebpy.nla import sptoeplitz

# %% Test 1.

col = np.array([1, 1, 2, 4, 5, 0, 0])
row = np.array([1, 3, 4, 0])
T = sptoeplitz(col, row)
print(csr_matrix.todense(T))

# %% Test 2.

n = 10
col = np.zeros(n)
row = np.zeros(2*n + 1)
row[int(n/2)] = 1
T = sptoeplitz(col, row)
print(csr_matrix.todense(T))