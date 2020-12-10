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
from matplotlib.pyplot import spy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from scipy.special import airy
import time

# Chebpy imports:
from chebpy.trig import trigpts, coeffs2vals, feval, vals2coeffs
from chebpy.trig import diffmat, multmat

# %% Solve on [0,2*pi].
    
# Grid:
n = 2000
x = trigpts(n, [0, 2*pi])

# Right-hand side f:
f = lambda x: 100 + 0*x

# Assemble matrices:
#start = time.time()
D1 = diffmat(n, 1, [0, 2*pi])
D2 = diffmat(n, 2, [0, 2*pi])
#end = time.time()
#print(f'Time (setup):  {end-start:.5f}s')
#start = time.time()
M0 = multmat(n, lambda x: 1000*np.cos(2*x), [0, 2*pi])
#end = time.time()
#print(f'Time (setup):  {end-start:.5f}s')
#start = time.time()
M1 = multmat(n, lambda x: np.sin(x), [0, 2*pi])
#end = time.time()
#print(f'Time (setup):  {end-start:.5f}s')
L = D2 + M1 @ D1 + M0
spy(L)

# Assemble RHS:
F = vals2coeffs(f(x))
#end = time.time()
#print(f'Time (setup):  {end-start:.5f}s')

# Sparse solve:
start = time.time()
U = spsolve(L, F)
end = time.time()
print(f'Time (solve):  {end-start:.5f}s')

# Plot and compute error:
u = coeffs2vals(U)
plt.figure()
plt.plot(x, u, '-')
# plt.plot(x, uex)
# error = np.max(np.abs(u - uex))/np.max(np.abs(uex))
# print(f'Error (L-inf): {error:.2e}')