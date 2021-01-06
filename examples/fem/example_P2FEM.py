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

# %% Solve -u''(x) + a*u(x) = f(x) on [-1,1] with u(-1) = u(1) = 0, P2 FEM.

# RHS and exact solution:
a = -4
nu = 2
f = lambda x: ((2*pi*nu)**2 + a)*np.sin(2*pi*nu*x)
uex = lambda x: np.sin(2*pi*nu*x)
    
# Seed random generator:
np.random.seed(1)

# Grid:
n = 20
xx = np.linspace(-1, 1, n + 2) # computation grid
xx[1:-1] += 1e-4*np.random.randn(n)
mm = np.zeros(n + 1)
for i in range(n + 1):
    mm[i] = (xx[i] + xx[i+1])/2

# Basis functions:
d1l = lambda i, x: (x > xx[i-1])*(x <= xx[i])
d1r = lambda i, x: (x > xx[i])*(x < xx[i+1])
V1l = lambda i, x: (x-mm[i-1])*(x-xx[i-1])/(xx[i]-mm[i-1])/(xx[i]-xx[i-1])
V1r = lambda i, x: (x-mm[i])*(x-xx[i+1])/(xx[i]-mm[i])/(xx[i]-xx[i+1])
V1 = lambda i, x: V1l(i, x)*d1l(i, x) + V1r(i, x)*d1r(i, x)
d2 = lambda i, x: (x > xx[i])*(x <= xx[i+1])
V2 = lambda i, x: (x-xx[i])*(x-xx[i+1])/(mm[i]-xx[i])/(mm[i]-xx[i+1])
V = lambda i, x: V1(i//2, x)*(i%2==0) + V2(i//2, x)*(i%2==1)*d2(i//2, x)
dV1l = lambda i, x: ((x-mm[i-1])+(x-xx[i-1]))/(xx[i]-mm[i-1])/(xx[i]-xx[i-1])
dV1r = lambda i, x: ((x-mm[i])+(x-xx[i+1]))/(xx[i]-mm[i])/(xx[i]-xx[i+1])
dV1 = lambda i, x: dV1l(i, x)*d1l(i, x) + dV1r(i, x)*d1r(i, x)
dV2 = lambda i, x: ((x-xx[i])+(x-xx[i+1]))/(mm[i]-xx[i])/(mm[i]-xx[i+1])
dV = lambda i, x: dV1(i//2, x)*(i%2==0) + dV2(i//2, x)*(i%2==1)*d2(i//2, x)

# Plot basis functions:
ss = np.linspace(-1, 1, 1000) # evalualtion grid
plt.figure()
for i in range(1, 2*(n + 1)):
    plt.plot(ss, V(i, ss))
plt.plot(xx, 0*xx, '.k')
plt.plot(mm, 0*mm, 'xk')
plt.figure()
for i in range(1, 2*(n + 1)):
    plt.plot(ss, dV(i, ss))
plt.plot(xx, 0*xx, '.k')
plt.plot(mm, 0*mm, 'xk')

# Assemble matrices:
start = time.time() 
K = np.zeros([2*n + 1, 2*n + 1])
M = np.zeros([2*n + 1, 2*n + 1])
for i in range(1, 2*(n + 1)):
    for j in range(1, 2*(n + 1)):
        if (np.abs(i - j) <= 2):
            for l in range(n + 1):
                I = min(i/2, j/2)
                J = max(i/2, j/2)
                if (np.abs(l + 1 - I) <= 1 and J <= l + 1):
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
F = np.zeros(2*n + 1)
for i in range(1, 2*(n + 1)):
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
    for i in range(1, 2*(n + 1)):
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