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
from chebpy.imex import start_imex
from chebpy.trig import multmat, trigpts
from chebpy.sph import coeffs2vals, feval, laplacian, vals2coeffs, polecond

# %% Solve u_t = alpha*Laplacian(u) + N(u) on the sphere.

# Initial condition:
x = lambda ll, tt: np.cos(ll) * np.sin(tt)
y = lambda ll, tt: np.sin(ll) * np.sin(tt)
z = lambda ll, tt: np.cos(tt)
u0 = lambda ll, tt: 5*x(ll,tt)*z(ll,tt) - 10*y(ll,tt)
    
# Nonlinearity and Laplacian constant:
alpha = 1e-2
c2v = lambda u: coeffs2vals(np.reshape(u, (n, n)))
v2c = lambda u: np.reshape(vals2coeffs(u), (n*n, 1))
Nv = lambda u: u - u**3
N = lambda u: v2c(Nv(c2v(u)))
    
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
U, NU = start_imex(N, n, dt, U0, alpha, q)
    
# Time-stepping:
start = time.time()
itrmax = round(T/dt)
for itr in range(q, itrmax + 1):
    
    # Compute new solution:
    b = 48*U[0] - 36*U[1] + 16*U[2] - 3*U[3] 
    b += dt*(48*NU[0] - 72*NU[1] + 48*NU[2] - 12*NU[3])
    Unew = lu.solve(Tsin2 @ b)
    
    # Update:
    U = [Unew] + U
    NU = [N(Unew)] + NU
    U.pop()
    NU.pop()
    
end = time.time()
print(f'Time   (solve): {end-start:.5f}s')

# Plot final solution:
U = np.reshape(U[0], (n, n)).T
u = coeffs2vals(U)
plt.figure()
plt.contourf(LAM, TT, u, 40, cmap=cm.coolwarm)
plt.colorbar()

# Pole condition:
P = polecond(U)
print(f'Pole condition: {P:.2e}')