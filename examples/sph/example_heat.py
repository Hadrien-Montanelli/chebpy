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
from scipy.sparse import csc_matrix
from scipy.sparse import eye
from scipy.sparse import kron
from scipy.sparse.linalg import splu
import time

# Chebpy imports:
from chebpy.trig import multmat, trigpts
from chebpy.sph import coeffs2vals, feval, laplacian, spharm, vals2coeffs

# %% Solve u_t = alpha*Laplacian(u) on the sphere.
    
# Initial condition:
l = 5
m = 3
u0 = spharm(l, m)

# Exact solution at t:
uex = lambda t, ll, tt: np.exp(-t)*u0(ll, tt)
    
# Laplacian constant:
alpha = 1/(l*(l + 1))
    
# Grid points:
n = 64
dom = [-pi, pi]
lam = trigpts(n, dom)
tt = trigpts(n, dom)
LAM, TT = np.meshgrid(lam, tt)

# Time discretization:
dt = 1e-3
T = 1
q = 4

# Plot initial condition:
plt.figure()
plt.contourf(LAM, TT, feval(u0, LAM, TT), 40, cmap=cm.coolwarm)
plt.colorbar()

# Assemble initial condition:
U0 = vals2coeffs(feval(u0, LAM, TT))
U0 = np.reshape(U0.T, (n*n, 1))

# Assemble Laplacian:
start = time.time()
L = alpha*laplacian(n, 0)

# LU factorization of IMEX-BDF4 matrix:
I = eye(n)
Tsin2 = multmat(n, lambda x: np.sin(x)**2, [-pi, pi])
Tsin2 = csc_matrix(kron(I, Tsin2))
lu = splu(25*Tsin2 - 12*dt*L)
end = time.time()
print(f'Time   (setup): {end-start:.5f}s')

# Start initial condition with exact solution:
U = [U0]
for i in range(1, q):
    t = i*dt
    U = [np.exp(-t)*U0] + U
    
# Time-stepping:
start = time.time()
itrmax = round(T/dt)
for itr in range(q, itrmax + 1):
    
    # Compute new solution:
    b = 48*U[0] - 36*U[1] + 16*U[2] - 3*U[3] 
    Unew = lu.solve(Tsin2 @ b)
    
    # Update:
    U = [Unew] + U
    U.pop()
end = time.time()
print(f'Time   (solve): {end-start:.5f}s')

# Plot final solution:
U = np.reshape(U[0], (n, n)).T
u = coeffs2vals(U)
plt.figure()
plt.contourf(LAM, TT, u, 40, cmap=cm.coolwarm)
plt.colorbar()

# Plot exact solution:
Uex = np.exp(-T)*np.reshape(U0, (n, n)).T
uex = coeffs2vals(Uex)
plt.figure()
plt.contourf(LAM, TT, uex, 40, cmap=cm.coolwarm)
plt.colorbar()

# Error:
error = np.max(np.abs(uex - u))/np.max(np.abs(uex))
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