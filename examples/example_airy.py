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
from scipy.special import airy

# Chebpy imports:
from chebpy import chebpts, coeffs2vals, feval, vals2coeffs
from chebpy import diffmat, multmat, spconvert

# %% Solve eps*u''(x) - x*u(x) = 0 with u(-1) = c1 and u(1) = c2.
    
# Scaled Airy functions:
eps = 1e-4
N = 1000
x = chebpts(N)
ai, aip, bi, bip = airy(eps**(-1/3)*x)
    
# Right-hand side f and boundary conditions c1 and c2:
f = lambda x: 0*x
c1 = ai[0]
c2 = ai[-1]

# Exact solution:
uex = ai  

# Assemble matrices:
D = diffmat(N, 2)
S0 = spconvert(N, 0)
S1 = spconvert(N, 1)
M = multmat(N, lambda x: -x)
L = eps*np.array(csr_matrix.todense(D)) + S1 @ S0 @ M
for k in range(N):
    T = np.zeros(N)
    T[k] = 1
    L[-2, k] = feval(T, -1)
    L[-1, k] = feval(T, 1)
L = np.roll(L, 2, axis=0)
L = csr_matrix(L)

# Assemble RHS:
F = vals2coeffs(f(x))
F = S1*S0*F
F = np.roll(F, 2)
F[0] = c1
F[1] = c2
F = csr_matrix(np.round(F, 15)).T

# Sparse solve:
U = spsolve(L, F)

# Plot and compute error:
u = coeffs2vals(U)
fig = plt.figure()
plt.plot(x, u, '.')
plt.plot(x, uex)
error = np.max(np.abs(u - uex))/np.max(np.abs(uex))
print('Error: (Airy)     ', error)