#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:37:10 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library import:
from matplotlib.pyplot import spy
import numpy as np

# Chebpy imports:
from chebpy import chebpts, coeffs2vals, multmat, spconvert, vals2coeffs

# %% Multiplication g = f*h on [-1,1].

# Functions:
f = lambda x: np.cos(x)*np.exp(x)
h = lambda x: np.sin(x)*x**2
gex = lambda x: f(x)*h(x)

# Grid:
n = 50
x = chebpts(n)

# Compute coeffs of f:
F = vals2coeffs(f(x))

# Multiplication by h matrix in coefficient space:
M = multmat(n, h)
spy(M)

# Multiply:
G = M @ F

# Convert to value space:
g = coeffs2vals(G)

# Error:
error = np.max(np.abs(g - gex(x)))/np.max(np.abs(gex(x)))
print('Error:', error)

# %% Multiplication g = f*h on [0,2].

# Functions:
f = lambda x: np.cos(x)*np.exp(x)
h = lambda x: np.sin(x)*x**2
gex = lambda x: f(x)*h(x)

# Grid:
n = 50
x = chebpts(n, [0, 2])

# Compute coeffs of f:
F = vals2coeffs(f(x))

# Multiplication by h matrix in coefficient space:
M = multmat(n, h, [0, 2])
spy(M)

# Multiply:
G = M @ F

# Convert to value space:
g = coeffs2vals(G)

# Error:
error = np.max(np.abs(g - gex(x)))/np.max(np.abs(gex(x)))
print('Error:', error)

# %% Multiplication g = f*h in the C^{(lam)} basis.

# Basis lambda:
lam = 3

# Functions:
f = lambda x: np.cos(x)*np.cos(2*x)
h = lambda x: np.sin(x)*x**10
gex = lambda x: f(x)*h(x)

# Grid:
n = 100
x = chebpts(n)

# Compute coeffs of f and convert to the C^{(lam)} basis:
F = vals2coeffs(f(x))
for k in range(lam):
    S = spconvert(n, k)
    F = S @ F
    
# Multiplication by h matrix in in the C^{(lam)} basis:
M = multmat(n, h, [-1, 1], lam)
spy(M)

# Multiply in the C^{(lam)} basis:
G = M @ F

# Exact coefficients in the C^{(lam)} basis:
Gex = vals2coeffs(gex(x))
for k in range(lam):
    S = spconvert(n, k)
    Gex = S @ Gex
    
# Error:
error = np.max(np.abs(G - Gex))/np.max(np.abs(Gex))
print('Error:', error)
