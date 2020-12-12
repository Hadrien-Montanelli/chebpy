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
from chebpy.trig import coeffs2vals, multmat, trigpts, vals2coeffs

# %% Multiplication g = f*h on [-1,1].

# Functions:
f = lambda x: np.cos(pi*x)
h = lambda x: np.exp(np.sin(10*pi*x))
gex = lambda x: f(x)*h(x)

# Grid:
n = 10000
x = trigpts(n)

# Compute coeffs of f:
F = vals2coeffs(f(x))

# Multiplication by h matrix in coefficient space:
start = time.time()
M = multmat(n, h)
end = time.time()
print(f'Time   (setup): {end-start:.5f}s')
plt.spy(M)

# Multiply:
start = time.time()
G = M @ F
end = time.time()
print(f'Time (product): {end-start:.5f}s')

# Convert to value space:
g = coeffs2vals(G)

# Error:
error = np.max(np.abs(g - gex(x)))/np.max(np.abs(gex(x)))
print(f'Error  (L-inf): {error:.2e}')

# %% Multiplication g = f*h on [0,2*pi].

# Functions:
f = lambda x: np.cos(x)
h = lambda x: np.exp(np.sin(10*x))
gex = lambda x: f(x)*h(x)

# Grid:
n = 10000
x = trigpts(n, [0, 2*pi])

# Compute coeffs of f:
F = vals2coeffs(f(x))

# Multiplication by h matrix in coefficient space:
start = time.time()
M = multmat(n, h, [0, 2*pi])
end = time.time()
print(f'Time   (setup): {end-start:.5f}s')
plt.figure()
plt.spy(M)

# Multiply:
start = time.time()
G = M @ F
end = time.time()
print(f'Time (product): {end-start:.5f}s')

# Convert to value space:
g = coeffs2vals(G)

# Error:
error = np.max(np.abs(g - gex(x)))/np.max(np.abs(gex(x)))
print(f'Error  (L-inf): {error:.2e}')