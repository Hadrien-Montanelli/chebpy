#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:29:37 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve
import time

# Chebpy imports:
from chebpy.trig import coeffs2vals, trigpts, vals2coeffs
from chebpy.trig import diffmat, multmat

# %% Solve u''(x) + sin(x)*u'(x) + 1000*cos(2x)*u(x) = f on [0,2*pi].
    
# Grid:
n = 10000
x = trigpts(n, [0, 2*pi])

# Right-hand side f:
f = lambda x: 100 + 0*x

# Assemble matrices:
start = time.time()
D1 = diffmat(n, 1, [0, 2*pi])
D2 = diffmat(n, 2, [0, 2*pi])
M0 = multmat(n, lambda x: 1000*np.cos(2*x), [0, 2*pi])
M1 = multmat(n, lambda x: np.sin(x), [0, 2*pi])
L = D2 + M1 @ D1 + M0
plt.figure()
plt.spy(L)

# Assemble RHS:
F = vals2coeffs(f(x))
end = time.time()
print(f'Time (setup): {end-start:.5f}s')

# Sparse solve:
start = time.time()
U = spsolve(L, F)
end = time.time()
print(f'Time (solve): {end-start:.5f}s')

# Plot and compute error:
u = coeffs2vals(U)
plt.figure()
plt.plot(x, u, '-')