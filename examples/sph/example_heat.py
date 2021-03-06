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
from scipy.sparse import csc_matrix, eye, kron
from scipy.sparse.linalg import splu
import time

# Chebpy imports:
from chebpy.imex import start_imex
from chebpy.trig import multmat, trigpts
from chebpy.sph import bmc, coeffs2vals, feval, vals2coeffs, pcond
from chebpy.sph import laplacian, spharm

# %% Solve u_t = alpha*Laplacian(u) on the sphere.
    
# Initial condition:
l = 5
m = 3
u0 = spharm(l, m)

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
L = alpha*laplacian(n)

# LU factorization of BDF4 matrix:
I = eye(n)
Tsin2 = multmat(n, lambda x: np.sin(x)**2, [-pi, pi])
Tsin2 = csc_matrix(kron(I, Tsin2))
lu = splu(25*Tsin2 - 12*dt*L)
end = time.time()
print(f'Time   (setup): {end-start:.5f}s')

# Start initial condition with LIRK4:
U, NU = start_imex(lambda u: 0*u, n, dt, U0, alpha, q)
    
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

# BMC symmetry:
S = bmc(U)
print(f'BMC-I symmetry: {S:.2e}')

# Pole condition:
P = pcond(U)
print(f'Pole condition: {P:.2e}')