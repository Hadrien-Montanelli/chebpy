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
from scipy.sparse.linalg import spsolve
from scipy.special import jv, yv
import time

# Chebpy imports:
from chebpy.cheb import chebpts, coeffs2vals, feval, vals2coeffs
from chebpy.cheb import diffmat, multmat, spconvert

# %% Solve a2(x)u''(x) + a1(x)u'(x) + a0(x)u(x) = f(x) on [a,b],
#
#          with a0(x) = 4*x^3*exp(x^2),
#               a1(x) = -(2*x^2 + 1),
#               a2(x) = x, 
#                f(x) = 0,
#
#          and   u(x0) = c, a <= x0 <= b,
#                u(x1) = d, a <= x1 <= b.
#
# Exact solution is a linear combination of scaled Bessel functions J1 and Y1.

# Domain:
a = 1.1
b = 2.6

# Grid:
n = 400
x = chebpts(n, [a, b])

# Boundary conditions:
x0 = a
x1 = b
c = 4.4
d = 1.2

# Variable coefficients:
a0 = lambda x: 4*x**3*np.exp(x**2)
a1 = lambda x: -(2*x**2 + 1)
a2 = lambda x: x
    
# Right-hand side:
f = lambda x: 0*x

# Exact solution:
z = 2*np.exp(x**2/2)
J1 = jv(1, z)
Y1 = yv(1, z)
C2 = (np.exp(-b**2/2)*d - np.exp(-a**2/2)*c*J1[-1]/J1[0])
C2 = C2/(Y1[-1] - Y1[0]*J1[-1]/J1[0])
C1 = (np.exp(-a**2/2)*c - C2*Y1[0])/J1[0]
uex = np.exp(x**2/2)*(C1*J1 + C2*Y1)

# Assemble matrices:
start = time.time()
D1 = diffmat(n, 1, [a, b])
D2 = diffmat(n, 2, [a, b])
S0 = spconvert(n, 0)
S1 = spconvert(n, 1)
M0 = multmat(n, a0, [a, b], 0)
M1 = multmat(n, a1, [a, b], 1)
M2 = multmat(n, a2, [a, b], 2)
L = M2 @ D2 + S1 @ M1 @ D1 + S1 @ S0 @ M0
L = np.array(csr_matrix.todense(L))
for k in range(n):
    T = np.zeros(n)
    T[k] = 1
    L[-2, k] = feval(T, 2/(b-a)*x0 - (a+b)/(b-a))
    L[-1, k] = feval(T, 2/(b-a)*x1 - (a+b)/(b-a))
L = np.roll(L, 2, axis=0)
L = csr_matrix(np.round(L, 15))
spy(L)

# Assemble RHS:
F = vals2coeffs(f(x))
F = S1 @ S0 @ F
F = np.roll(F, 2)
F[0] = c
F[1] = d
F = csr_matrix(np.round(F, 13)).T
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