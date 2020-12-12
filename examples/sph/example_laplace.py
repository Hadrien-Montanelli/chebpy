#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:23:46 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library import:
from math import pi
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import block_diag, eye, kron, lil_matrix
from scipy.sparse.linalg import spsolve
import time

# Chebpy imports:
from chebpy.trig import coeffs2vals, trigpts, vals2coeffs
from chebpy.trig import diffmat, multmat

# %% Solve u_xx + u_yy = f on the sphere.

# DFS method:
def feval(f, LAM, TT):
    n = len(LAM)
    Y = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if (TT[i, 0] >= 0):
                Y[i, j] = f(LAM[0, j], TT[i, 0])
            else:
                if (LAM[0, j] <=0):
                    Y[i, j] = f(LAM[0, j] + pi, -TT[i, 0])
                else:
                    Y[i, j] = f(LAM[0, j] - pi, -TT[i, 0])
    return Y
    
# RHS:
l = 4
f1 = lambda lam, tt: l*(l+1) * np.sin(tt)**l * np.cos(l*lam)
f2 = lambda lam, tt: (l+1)*(l+2) * np.cos(tt) * np.sin(tt)**l * np.cos(l*lam)
f = lambda lam, tt: f1(lam, tt) + f2(lam, tt)
    
# Exact solution:
uex1 = lambda lam, tt: -np.sin(tt)**l * np.cos(l*lam)
uex2 = lambda lam, tt: -np.cos(tt) * np.sin(tt)**l * np.cos(l*lam)
uex = lambda lam, tt: uex1(lam, tt) + uex2(lam, tt)
    
# Grid points:
n = 64
dom = [-pi, pi]
lam = trigpts(n, dom)
tt = trigpts(n, dom)
LAM, TT = np.meshgrid(lam, tt)

# Assemble RHS:
F = vals2coeffs(vals2coeffs(feval(f, LAM, TT)).T).T 
F = np.reshape(F.T, (n*n, 1))

# Zero-mean condition:
F[n*int(n/2) + int(n/2)] = 0

# Assemble Laplacian:
start = time.time()
I = eye(n)
D1 = diffmat(n, 1, dom)
D2 = diffmat(n, 2, dom)
Tsin2 = multmat(n, lambda x: np.sin(x)**2, dom)
Tcossin = multmat(n, lambda x: np.cos(x)*np.sin(x), dom)
blocks = []
for k in range(n):
    block = Tsin2 @ D2 + Tcossin @ D1 + D2[k, k]*I
    if (k == int(n/2)):
        mm = np.arange(-n/2, n/2)
        mm[int(n/2)-1] = 0
        mm[int(n/2)+1] = 0
        en = 2*pi*(1 + np.exp(1j*pi*mm))/(1 - mm**2)
        en[int(n/2)-1] = 0
        en[int(n/2)+1] = 0
        block = lil_matrix(block)
        block[int(n/2), :] = en
    blocks.append(block)
L = block_diag(blocks, format='csr')
end = time.time()
print(f'Time   (setup): {end-start:.5f}s')
plt.figure()
plt.spy(L)

# Sparse solve:
start = time.time()
U = spsolve(L, kron(I, Tsin2) @ F)
end = time.time()
print(f'Time   (solve): {end-start:.5f}s')
U = np.reshape(U, (n, n)).T

# Plot solution:
u = coeffs2vals(coeffs2vals(U).T).T
plt.figure()
plt.contourf(LAM, TT, u, 40, cmap=cm.coolwarm)
plt.colorbar()

# Plot exact solution:
plt.figure()
plt.contourf(LAM, TT, feval(uex, LAM, TT), 40, cmap=cm.coolwarm)
plt.colorbar()

# Error:
error = feval(uex, LAM, TT) - u
error = np.max(np.abs(error))/np.max(np.abs(feval(uex, LAM, TT)))
print(f'Error  (L-inf): {error:.2e}')

# Pole condition:
rowsum1 = np.zeros(n)
rowsum2 = np.zeros(n)
for i in range(n):
    if (i == n/2+1):
        continue
    else:
        rowsum1[i] = np.abs(np.sum(U[:, i]))
        rowsum2[i] = np.abs(np.sum(U[:, i] * (-1)**(i-n/2-1)))
P = max(np.max(rowsum1), np.max(rowsum2))
print(f'Pole condition: {P:.2e}')