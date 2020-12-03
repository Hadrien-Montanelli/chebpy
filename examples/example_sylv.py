#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 17:51:05 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library import:
import numpy as np

# Chebpy imports:
from chebpy import gensylv
from chebpy import sylv

# %% Solve Sylvester equation AX - XB = C.

m = 5
n = 6
A = np.ones([m, m]) + np.random.randn(m, m)
B = np.eye(n) + 3*np.ones([n, n])
C = np.random.randn(m, n)

X = sylv(A, B, C)
error = np.linalg.norm(A @ X - X @ B - C)
print('Error:', error)

# %% Solve generalized Sylvester equation AXB^T + CXD^T = E.

#np.random.seed(2)
m = 5
n = 6
A = 4*np.ones([m, m]) + np.random.randn(m, m)
B = 2*np.eye(n) + np.random.randn(n, n)
C = 3*np.eye(m) + np.random.randn(m, m)
D = 5*np.eye(n) + np.random.randn(n, n)
E = np.random.randn(m, n)

X = gensylv(A, B, C, D, E)
error = np.linalg.norm(A @ X @ B.T + C @ X @ D.T - E)
print('Error:', error)