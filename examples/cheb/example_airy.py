#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:29:37 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
import matplotlib.pyplot as plt
from matplotlib.pyplot import spy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import airy
import time

# Chebpy imports:
from chebpy.cheb import chebpts, coeffs2vals, feval, vals2coeffs
from chebpy.cheb import diffmat, multmat, spconvert

# %% Solve eps*u''(x) - x*u(x) = 0 on [-1,1] with u(-1) = c and u(1) = d.
    
# Grid:
n = 100000
x = chebpts(n)

# Scaled Airy functions:
eps = 1e-4
ai, aip, bi, bip = airy(eps**(-1/3)*x)

# Boundary conditions:
c = ai[0]
d = ai[-1]   

# Right-hand side:
f = lambda x: 0*x

# Exact solution:
uex = ai  

# Assemble matrices:
start = time.time()
D2 = diffmat(n, 2)
S0 = spconvert(n, 0)
S1 = spconvert(n, 1)
M = multmat(n, lambda x: -x)
L = eps*D2 + S1 @ S0 @ M
L = lil_matrix(L)
L[-2, :] = (-1)**np.arange(0, n)
L[-1, :] = np.ones(n)
L = csr_matrix(L)
spy(L)

# Assemble RHS:
F = vals2coeffs(f(x))
F = S1 @ S0 @ F
F[-2] = c
F[-1] = d
end = time.time()
print(f'Time  (setup):  {end-start:.5f}s')

# Sparse solve:
start = time.time()
U = spsolve(L, F)
end = time.time()
print(f'Time  (solve):  {end-start:.5f}s')

# Plot and compute error:
u = coeffs2vals(U)
plt.figure()
plt.plot(x, u, '.')
plt.plot(x, uex)
error = np.max(np.abs(u - uex))/np.max(np.abs(uex))
print(f'Error (L-inf): {error:.2e}')

# %% Solve eps*u''(x) - x*u(x) = 0 on [a,b] with u(a) = c and u(b) = d.
    
# Domain:
a = -2
b = 0

# Grid:
n = 1000
x = chebpts(n, [a, b])

# Scaled Airy functions:
eps = 1e-4
ai, aip, bi, bip = airy(eps**(-1/3)*x)
    
# Boundary conditions:
c = ai[0]
d = ai[-1]
 
# Right-hand side f:
f = lambda x: 0*x

# Exact solution:
uex = ai  

# Assemble matrices: O(n^2) complexity because of feval.
start = time.time()
D2 = diffmat(n, 2, [a, b])
S0 = spconvert(n, 0)
S1 = spconvert(n, 1)
M = multmat(n, lambda x: -x, [a, b])
L = eps*D2 + S1 @ S0 @ M
L = lil_matrix(L)
for k in range(n):
    T = np.zeros(n)
    T[k] = 1
    L[-2, k] = feval(T, 2/(b-a)*a - (a+b)/(b-a)) 
    L[-1, k] = feval(T, 2/(b-a)*b - (a+b)/(b-a))
L = csr_matrix(L)
plt.figure()
spy(L)

# Assemble RHS:
F = vals2coeffs(f(x))
F = S1 @ S0 @ F
F[-2] = c
F[-1] = d
end = time.time()
print(f'Time  (setup): {end-start:.5f}s')

# Sparse solve:
start = time.time()
U = spsolve(L, F)
end = time.time()
print(f'Time  (solve): {end-start:.5f}s')

# Plot and compute error:
u = coeffs2vals(U)
plt.figure()
plt.plot(x, u, '.')
plt.plot(x, uex)
error = np.max(np.abs(u - uex))/np.max(np.abs(uex))
print(f'Error (L-inf): {error:.2e}')