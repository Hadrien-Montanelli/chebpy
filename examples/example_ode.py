#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:29:37 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library import:
import matplotlib.pylab as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Chebpy imports:
from chebpy import chebpts, coeffs2vals, vals2coeffs
from chebpy import diffmat, spconvert

# %% Solve u'(x) = f(x) on [-1,1] with u(-1) = c.

# Right-hand side f and boundary condition c:
f = lambda x: np.cos(x)
c = 4

# Exact solution:
uex = lambda x: np.sin(x) + c - np.sin(-1)    

# Assemble matrices:
N = 20
xx = chebpts(N)
D = diffmat(N, 1)
S = spconvert(N, 0)
L = D.copy()
L = np.array(csr_matrix.todense(L))
for k in range(N):
    L[-1, k] = (-1)**k
L = np.roll(L, 1, axis=0)
L = csr_matrix(L)

# Assemble RHS:
F = vals2coeffs(f(xx))
F = S*F
F[-1] = c
F = np.roll(F, 1)
F = csr_matrix(F).T

# Sparse solve:
U = spsolve(L, F)

# Plot and compute error:
u = coeffs2vals(U)
plt.plot(xx, u, '.-')
plt.plot(xx, uex(xx))
error = u - uex(xx)
print('Error:', np.max(np.abs(error)))