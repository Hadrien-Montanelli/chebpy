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
from chebpy import chebpts, coeffs2vals, multmat, vals2coeffs

# %% Multiplication g = f*h:

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