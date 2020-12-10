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
from matplotlib.pyplot import spy
import numpy as np

# Chebpy imports:
from chebpy.trig import trigpts, coeffs2vals, multmat, vals2coeffs

# %% Multiplication g = f*h on [-1,1].

# Functions:
f = lambda x: np.cos(pi*x)
h = lambda x: np.cos(np.exp(np.sin(10*pi*x)))
gex = lambda x: f(x)*h(x)

# Grid:
n = 600
x = trigpts(n)

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

# %% Multiplication g = f*h on [0,2*pi].

# Functions:
f = lambda x: np.cos(x)
h = lambda x: np.cos(np.exp(np.sin(10*x)))
gex = lambda x: f(x)*h(x)

# Grid:
n = 600
x = trigpts(n, [0, 2*pi])

# Compute coeffs of f:
F = vals2coeffs(f(x))

# Multiplication by h matrix in coefficient space:
M = multmat(n, h, [0, 2*pi])
plt.figure()
spy(M)

# Multiply:
G = M @ F

# Convert to value space:
g = coeffs2vals(G)

# Error:
error = np.max(np.abs(g - gex(x)))/np.max(np.abs(gex(x)))
print('Error:', error)