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
from scipy.special import jv, yv

# Chebpy imports:
from chebpy import chebpts, coeffs2vals, feval, vals2coeffs
from chebpy import diffmat, multmat, spconvert

# %% Solve a2(x)u''(x) + a1(x)u'(x) + a0(x)u(x) = f(x) on [a,b],
#
#          with a0(x) = 4*x^3*exp(x^2),
#               a1(x) = -(2*x^2 + 1),
#               a2(x) = x, 
#                f(x) = 0,
#
#          and   u(a) = c,
#                u(b) = d.
#
# Exact solution is a linear combination of scaled Bessel functions J1 and Y1.

# Domain and boundary conditions:
a = 1.1
b = 2.5
c = 4.4
d = 1.2

# Grid:
n = 300
x = chebpts(n, [a, b])

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
D1 = diffmat(n, 1, [a, b])
D2 = diffmat(n, 2, [a, b])
S0 = spconvert(n, 0)
S1 = spconvert(n, 1)
M0 = multmat(n, a0, [a, b])
M1 = multmat(n, a1, [a, b], 1)
M2 = multmat(n, a2, [a, b], 2)
L = M2 @ np.array(csr_matrix.todense(D2)) + S1 @ M1 @ D1 + S1 @ S0 @ M0
for k in range(n):
    T = np.zeros(n)
    T[k] = 1
    L[-2, k] = feval(T, 2/(b-a)*a - (a+b)/(b-a))
    L[-1, k] = feval(T, 2/(b-a)*b - (a+b)/(b-a))
L = np.roll(L, 2, axis=0)
L = csr_matrix(L)

# Assemble RHS:
F = vals2coeffs(f(x))
F = S1 @ S0 @ F
F = np.roll(F, 2)
F[0] = c
F[1] = d
F = csr_matrix(np.round(F, 15)).T

# Sparse solve:
U = spsolve(L, F)

# Plot and compute error:
u = coeffs2vals(U)
fig = plt.figure()
plt.plot(x, u, '.')
plt.plot(x, uex)
error = np.max(np.abs(u - uex))/np.max(np.abs(uex))
print('Error:', error)