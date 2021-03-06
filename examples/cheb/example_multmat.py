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
from chebpy.cheb import chebpts, coeffs2vals, multmat, spconvert, vals2coeffs

# %% Multiplication g = f*h on [-1,1].

# Functions:
f = lambda x: np.cos(x)*np.exp(x)
h = lambda x: np.sin(x)*x**2
gex = lambda x: f(x)*h(x)

# Grid:
n = 10000
x = chebpts(n)

# Compute coeffs of f:
F = vals2coeffs(f(x))

# Multiplication by h matrix in coefficient space:
start = time.time()
M = multmat(n, h)
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

# %% Multiplication g = f*h on [0,2].

# Functions:
f = lambda x: np.cos(x)*np.exp(x)
h = lambda x: np.sin(x)*x**2
gex = lambda x: f(x)*h(x)

# Grid:
n = 10000
x = chebpts(n, [0, 2])

# Compute coeffs of f:
F = vals2coeffs(f(x))

# Multiplication by h matrix in coefficient space:
start = time.time()
M = multmat(n, h, [0, 2])
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

# %% Multiplication g = f*h in the C^{(lam)} basis.

# Basis lambda:
lam = 2

# Functions:
f = lambda x: np.cos(x)*np.exp(x)
h = lambda x: np.sin(x)*x**2
gex = lambda x: f(x)*h(x)

# Grid:
n = 400
x = chebpts(n)

# Compute coeffs of f and convert to the C^{(lam)} basis:
F = vals2coeffs(f(x))
for k in range(lam):
    S = spconvert(n, k)
    F = S @ F
    
# Multiplication by h matrix in in the C^{(lam)} basis:
start = time.time()
M = multmat(n, h, [-1, 1], lam)
end = time.time()
print(f'Time   (setup): {end-start:.5f}s')
plt.figure()
plt.spy(M)

# Multiply in the C^{(lam)} basis:
start = time.time()
G = M @ F
end = time.time()
print(f'Time (product): {end-start:.5f}s')

# Exact coefficients in the C^{(lam)} basis:
Gex = vals2coeffs(gex(x))
for k in range(lam):
    S = spconvert(n, k)
    Gex = S @ Gex
    
# Error:
error = np.max(np.abs(G - Gex))/np.max(np.abs(Gex))
print(f'Error  (L-inf): {error:.2e}')