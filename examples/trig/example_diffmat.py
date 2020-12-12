#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:37:10 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import time

# Chebpy imports:
from chebpy.trig import coeffs2vals, diffmat, trigpts, vals2coeffs

# %% First-order differentiation on [-pi, pi].

# Function:
w = 100
f = lambda x: np.cos(w*x)
dfex = lambda x: -w*np.sin(w*x)

# Grid:
n = 4*w
dom = [-pi, pi]
x = trigpts(n, dom)

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

# Convert to value space:
df = coeffs2vals(DF)

# Error:
error = np.max(np.abs(df - dfex(x)))/np.max(np.abs(dfex(x)))
print(f'Error  (L-inf): {error:.2e}')


# %% Second-order differentiation on [-pi, pi].

# Function:
w = 100
f = lambda x: np.cos(w*x)
d2fex = lambda x: -w**2*np.cos(w*x)

# Grid:
n = 4*w
dom = [-pi, pi]
x = trigpts(n, dom)

# Compute coeffs of f:
F = vals2coeffs(f(x))

# Differentiation matrix in coefficient space:
start = time.time()
D2 = diffmat(n, 2, dom)
end = time.time()
print(f'Time   (setup): {end-start:.5f}s')
plt.figure()
plt.spy(D)

# Differentiate:
start = time.time()
D2F = D2 @ F
end = time.time()
print(f'Time (product): {end-start:.5f}s')

# Convert to value space:
d2f = coeffs2vals(D2F)

# Error:
error = np.max(np.abs(d2f - d2fex(x)))/np.max(np.abs(d2fex(x)))
print(f'Error  (L-inf): {error:.2e}')