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
from scipy.sparse import eye, kron
from scipy.sparse.linalg import spsolve
import time

# Chebpy imports:
from chebpy.trig import multmat, trigpts
from chebpy.sph import coeffs2vals, feval, laplacian, vals2coeffs, polecond

# %% Solve Laplacian(u) = f on the sphere.

# RHS:
l = 10
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
F = vals2coeffs(feval(f, LAM, TT))
F = np.reshape(F.T, (n*n, 1))

# Zero-mean condition:
F[n*int(n/2) + int(n/2)] = 0

# Assemble Laplacian (with mean condition):
start = time.time()
L = laplacian(n, 1)
end = time.time()
print(f'Time   (setup): {end-start:.5f}s')
plt.figure()
plt.spy(L)

# Sparse solve:
start = time.time()
I = eye(n)
Tsin2 = multmat(n, lambda x: np.sin(x)**2, dom)
U = spsolve(L, kron(I, Tsin2) @ F)
end = time.time()
print(f'Time   (solve): {end-start:.5f}s')

# Plot solution:
U = np.reshape(U, (n, n)).T
u = coeffs2vals(U)
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
P = polecond(U)
print(f'Pole condition: {P:.2e}')