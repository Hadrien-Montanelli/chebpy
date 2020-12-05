#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:23:46 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library import:
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import time

# Chebpy imports:
from chebpy import chebpts, coeffs2vals, feval, vals2coeffs
from chebpy import diffmat, gensylv, multmat, spconvert

# %% Solve r*u_rr - u_r + r*u_zz = f on [ra,rb]x[za,zb], Dirichlet conditions.

# Domain:
ra = 0.5
rb = 1.5
za = -0.6
zb = 0.6
r0 = ra
r1 = rb
z0 = za
z1 = zb

# RHS:
f0 = 1
R0 = 1.1
a = 0.5
f = lambda r, z: -f0*r**3 - f0*R0**2*r

# Exact solution:
scl = f0*a**2*R0**2/2
uex = lambda r, z: scl*(1 - z**2/a**2 -((r-R0)/a + (r-R0)**2/(2*a*R0))**2)
    
# Boundary condtions:
g1 = lambda z: uex(r0, z) # u(r0, z) = g1(z)
g2 = lambda z: uex(r1, z) # u(r1, z) = g2(z)
h1 = lambda r: uex(r, z0) # u(r, z0) = h1(r)
h2 = lambda r: uex(r, z1) # u(r, z1) = h2(r)

# Grid points:
n = 100
r = chebpts(n, [ra, rb])
z = chebpts(n, [za, zb])
R, Z = np.meshgrid(r, z)

# Assemble differentiation matrices:
start = time.time()
S0 = np.array(csr_matrix.todense(spconvert(n, 0)))
S1 = np.array(csr_matrix.todense(spconvert(n, 1)))
D1r = np.array(csr_matrix.todense(diffmat(n, 1, [ra, rb])))
D2r = np.array(csr_matrix.todense(diffmat(n, 2, [ra, rb])))
D2z = np.array(csr_matrix.todense(diffmat(n, 2, [za, zb])))
M0 = multmat(n, lambda r: r, [ra, rb], 0)
M2 = multmat(n, lambda r: r, [ra, rb], 2)
A1 = S1 @ S0 @ np.eye(n)
C1 = M2 @ D2r - S1 @ D1r
A2 = D2z
C2 = S1 @ S0 @ M0

# Assemble boundary conditions:
Bx = np.zeros([2, n])
By = np.zeros([2, n])
G = np.zeros([2, n])
H = np.zeros([2, n])
for k in range(n):
    T = np.zeros(n)
    T[k] = 1
    Bx[0, k] = feval(T, 2/(zb-za)*z0 - (za+zb)/(zb-za))
    By[0, k] = feval(T, 2/(rb-ra)*r0 - (ra+rb)/(rb-ra))
    Bx[1, k] = feval(T, 2/(zb-za)*z1 - (za+zb)/(zb-za))
    By[1, k] = feval(T, 2/(rb-ra)*r1 - (ra+rb)/(rb-ra))
G[0, :] = vals2coeffs(g1(z))
G[1, :] = vals2coeffs(g2(z))
H[0, :] = vals2coeffs(h1(r))
H[1, :] = vals2coeffs(h2(r))
Bx_hat = Bx[0:2, 0:2]
Bx = np.linalg.inv(Bx_hat) @ Bx
G = np.linalg.inv(Bx_hat) @ G
By_hat = By[0:2, 0:2]
By = np.linalg.inv(By_hat) @ By
H = np.linalg.inv(By_hat) @ H

# Assemble right-hand side:
F = vals2coeffs(vals2coeffs(f(R, Z)).T).T
F = (S1 @ S0) @ F @ (S1 @ S0).T

# Assemble matrices for the generalized Sylvester equation:
Ft = F - A1[:n, :2] @ H @ C1.T - (A1 - A1[:n, :2] @ By) @ G.T @ C1[:n, :2].T
Ft = Ft - A2[:n, :2] @ H @ C2.T - (A2 - A2[:n, :2] @ By) @ G.T @ C2[:n, :2].T
A1t = A1 - A1[:n, :2] @ By
A2t = A2 - A2[:n, :2] @ By
C1t = C1 - C1[:n, :2] @ Bx
C2t = C2 - C2[:n, :2] @ Bx
end = time.time()
print(f'Time  (setup): {end-start:.5f}s')

# Solve the generalized Sylvester equation:
A1t = A1t[:n-2, 2:] 
C1t = C1t[:n-2, 2:] 
A2t = A2t[:n-2, 2:]
C2t = C2t[:n-2:, 2:]
Ft = Ft[:n-2, :n-2]
start = time.time()
U22 = gensylv(A1t, C1t, A2t, C2t, Ft)
end = time.time()
print(f'Time  (solve): {end-start:.5f}s')

# Assemble solution:
U12 = H[:, 2:] - By[:, 2:] @ U22
U21 = G[:, 2:].T - U22 @ Bx[:, 2:].T
U11 = H[:, :2] - By[:, 2:] @ U21
U1 = np.concatenate((U11, U12), axis=1)
U2 = np.concatenate((U21, U22), axis=1)
U = np.concatenate((U1, U2), axis=0)

# Plot solution:
u = coeffs2vals(coeffs2vals(U).T).T
plt.contourf(R, Z, u, 40, cmap=cm.coolwarm)
plt.colorbar()

# Plot exact solution:
fig = plt.figure()
plt.contourf(R, Z, uex(R, Z), 40, cmap=cm.coolwarm)
plt.colorbar()

# Error:
error = np.max(np.abs(uex(R, Z) - u))/np.max(np.abs(uex(R, Z)))
print(f'Error (L-inf): {error:.2e}')