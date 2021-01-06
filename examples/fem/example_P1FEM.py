#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 12:01:48 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard imports:
from math import pi
import matplotlib.pyplot as plt
import numpy as np    
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import time

# Chebpy imports:
from chebpy.cheb import chebpts, quadwts

# %% Solve -u''(x) + a*u(x) = f(x) on [-1,1] with u(-1) = u(1) = 0, P1 FEM.

# RHS and exact solution:
a = -4
nu = 2
f = lambda x: ((2*pi*nu)**2 + a)*np.sin(2*pi*nu*x)
uex = lambda x: np.sin(2*pi*nu*x)
    
# Seed random generator:
np.random.seed(1)

# Grid:
n = 30
xx = np.linspace(-1, 1, n + 2) # computation grid
xx[1:-1] += 1e-4*np.random.randn(n)

# Basis functions:
dl = lambda i, x: (x > xx[i-1])*(x <= xx[i])
dr = lambda i, x: (x > xx[i])*(x < xx[i+1])
Vl = lambda i, x: (x - xx[i-1])/(xx[i] - xx[i-1])
Vr = lambda i, x: (xx[i+1] - x)/(xx[i+1] - xx[i])*dr(i, x)
V = lambda i, x: Vl(i, x)*dl(i, x) + Vr(i, x)*dr(i, x)
dVl = lambda i, x: 1/(xx[i] - xx[i-1])
dVr = lambda i, x: -1/(xx[i+1] - xx[i])
dV = lambda i, x: dVl(i, x)*dl(i, x) + dVr(i, x)*dr(i, x)

# Plot basis functions:
ss = np.linspace(-1, 1, 1000) # evalualtion grid
plt.figure()
for i in range(1, n + 1):
    plt.plot(ss, V(i, ss))
plt.plot(xx, 0*xx, '.k')
plt.figure()
for i in range(1, n + 1):
    plt.plot(ss, dV(i, ss))
plt.plot(xx, 0*xx, '.k')

# Assemble matrices:
start = time.time() 
K = np.zeros([n, n])
M = np.zeros([n, n])
for i in range(1, n + 1):
    for j in range(1, n + 1):
        if (np.abs(i - j) <= 1):
            for l in range(n + 1):
                if (np.abs(l + 1 - min(i, j)) <= 1 and max(i, j) <= l + 1):
                    dom = [xx[l], xx[l+1]]
                    N = 1000
                    x = chebpts(N, dom) # quadrature grid
                    w = quadwts(N, dom)
                    K[i-1, j-1] += w @ (dV(i, x) * dV(j, x))
                    M[i-1, j-1] += w @ (V(i, x) * V(j, x))
K = csr_matrix(K)
M = a*csr_matrix(M)
L = K + M

# Assemble RHS:
F = np.zeros(n)
for i in range(1, n + 1):
    for l in range(n + 1):
        dom = [xx[l], xx[l+1]]
        x = chebpts(N, dom)
        w = quadwts(N, dom)
        F[i-1] += w @ (f(x) * V(i, x))
end = time.time()
print(f'Time  (setup): {end-start:.5f}s')

# Sparse solve:
start = time.time() 
U = spsolve(L, F)
end = time.time()
print(f'Time  (solve): {end-start:.5f}s')

# Evaluate solution:
start = time.time() 
u = np.zeros(len(ss))
for k in range(len(ss)):
    for i in range(1, n + 1):
        u[k] += U[i-1] * V(i, ss[k])
end = time.time()    
print(f'Time  (feval): {end-start:.5f}s')

# Compute error:
error = np.max(np.abs(uex(ss) - u))/np.max(np.abs(uex(ss)))
print(f'Error (L-inf): {error:.2e}')

# Plot:
plt.figure()
plt.plot(ss, u)
plt.plot(ss, uex(ss), '--')