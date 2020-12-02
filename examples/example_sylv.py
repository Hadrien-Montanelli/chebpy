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

# %% Solve generalized Sylvester equation A1XB1 - A2XB2 = C.

m = 5
n = 5
A1 = np.ones([m, m]) + np.random.randn(m, m)
B1 = np.eye(n) + 3*np.ones([n, n])
A2 = 3*np.ones([m, m]) + np.random.randn(m, m)
B2 = 2*np.eye(n) + np.ones([n, n])
C = np.random.randn(m, n)

X = gensylv(A1, B1, A2, B2, C)
error = np.linalg.norm(A1 @ X @ B1 - A2 @ X @ B2 - C)
print('Error:', error)