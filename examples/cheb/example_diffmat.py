#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:37:10 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
import matplotlib.pyplot as plt
import numpy as np
import time

# Chebpy imports:
from chebpy.cheb import chebpts, diffmat, spconvert, vals2coeffs

# %% First-order differentiation on [-1, 1].

# Function:
w = 100
f = lambda x: np.cos(w*x)
dfex = lambda x: -w*np.sin(w*x)

# Grid:
n = 4*w
dom = [-1, 1]
x = chebpts(n, dom)

# Compute coeffs of f:
F = vals2coeffs(f(x))

# Differentiation matrix in coefficient space:
start = time.time()
D = diffmat(n, 1, dom)
end = time.time()
print(f'Time   (setup): {end-start:.5f}s')
plt.figure()
plt.spy(D)

# Differentiate:
start = time.time()
DF = D @ F
end = time.time()
print(f'Time (product): {end-start:.5f}s')

# Exact coefficients
S0 = spconvert(n, 0)
DFex = S0 @ vals2coeffs(dfex(x))

# Error:
error = np.max(np.abs(DF - DFex))/np.max(np.abs(DFex))
print(f'Error  (L-inf): {error:.2e}')


# %% Second-order differentiation on [0, 2].

# Function:
w = 100
f = lambda x: np.cos(w*x)
d2fex = lambda x: -w**2*np.cos(w*x)

# Grid:
n = 4*w
dom = [0, 2]
x = chebpts(n, dom)

# Compute coeffs of f:
F = vals2coeffs(f(x))

# Differentiation matrix in coefficient space:
start = time.time()
D2 = diffmat(n, 2, dom)
end = time.time()
print(f'Time   (setup): {end-start:.5f}s')
plt.figure()
plt.spy(D2)

# Differentiate:
start = time.time()
D2F = D2 @ F
end = time.time()
print(f'Time (product): {end-start:.5f}s')

# Exact coefficients
S0 = spconvert(n, 0)
S1 = spconvert(n, 1)
D2Fex = S1 @ S0 @ vals2coeffs(d2fex(x))

# Error:
error = np.max(np.abs(D2F - D2Fex))/np.max(np.abs(D2Fex))
print(f'Error  (L-inf): {error:.2e}')