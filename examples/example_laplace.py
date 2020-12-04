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
from scipy.sparse import csr_matrix

# Chebpy imports:
from chebpy import chebpts, coeffs2vals, feval, vals2coeffs
from chebpy import diffmat, gensylv, spconvert

# %% Solve u_xx + u_yy = f on [-1,1]x[-1,1] with Dirichlet boundary conditions.

# RHS:
w = 4*pi
f = lambda x,y: -2*w**2*np.sin(w*x)*np.sin(w*y)

# Boundary condtions:
g1 = lambda y: 0*y # u(-1, y) = g1(y)
g2 = lambda y: 0*y # u(+1, y) = g2(y)
h1 = lambda x: 0*x # u(x, -1) = h1(x)
h2 = lambda x: 0*x # u(x, +1) = h2(x)

# Exact solution:
uex = lambda x,y: np.sin(w*x)*np.sin(w*y)

# Grid points:
n = 100
x = chebpts(n)
y = chebpts(n)
X, Y = np.meshgrid(x, y)

# Assemble differentiation matrices:
S0 = spconvert(n, 0)
S1 = spconvert(n, 1)
A1 = S1 @ S0 @ np.eye(n)
C1 = np.array(csr_matrix.todense(diffmat(n, 2)))
A2 = np.array(csr_matrix.todense(diffmat(n, 2)))
C2 = S1 @ S0 @ np.eye(n)

# Assemble boundary conditions:
Bx = np.zeros([2, n])
By = np.zeros([2, n])
G = np.zeros([2, n])
H = np.zeros([2, n])
for k in range(n):
    T = np.zeros(n)
    T[k] = 1
    Bx[0, k] = feval(T, -1)
    By[0, k] = feval(T, -1)
    Bx[1, k] = feval(T, 1)
    By[1, k] = feval(T, 1)
G[0, :] = g1(y)
G[1, :] = g2(y)
H[0, :] = h1(x)
H[1, :] = h2(x)
Bx_hat = Bx[0:2, 0:2]
Bx = np.linalg.inv(Bx_hat) @ Bx
G = np.linalg.inv(Bx_hat) @ G
By_hat = By[0:2, 0:2]
By = np.linalg.inv(By_hat) @ By
H = np.linalg.inv(By_hat) @ H

# Assemble right-hand side:
F = vals2coeffs(vals2coeffs(f(X, Y)).T).T
F = (S1 @ S0) @ F @ (S1 @ S0).T

# Assemble matrices for the generalized Sylvester equation:
Ft = F - A1[:n, :2] @ H @ C1.T - (A1 - A1[:n, :2] @ By) @ G.T @ C1[:n, :2].T
Ft = Ft - A2[:n, :2] @ H @ C2.T - (A2 - A2[:n, :2] @ By) @ G.T @ C2[:n, :2].T
A1t = A1 - A1[:n, :2] @ By
A2t = A2 - A2[:n, :2] @ By
C1t = C1 - C1[:n, :2] @ Bx
C2t = C2 - C2[:n, :2] @ Bx

# Solve the generalized Sylvester equation:
A1t = A1t[:n-2, 2:] 
C1t = C1t[:n-2, 2:] 
A2t = A2t[:n-2, 2:]
C2t = C2t[:n-2:, 2:]
Ft = Ft[:n-2, :n-2]
U22 = gensylv(A1t, C1t, A2t, C2t, Ft)

# Assemble solution:
U12 = H[:, 2:] - By[:, 2:] @ U22
U21 = G[:, 2:].T - U22 @ Bx[:, 2:].T
U11 = H[:, :2] - By[:, 2:] @ U21
U1 = np.concatenate((U11, U12), axis=1)
U2 = np.concatenate((U21, U22), axis=1)
U = np.concatenate((U1, U2), axis=0)

# Plot solution:
u = coeffs2vals(coeffs2vals(U).T).T
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, u, cmap=cm.coolwarm, linewidth=0)
fig.colorbar(surf, shrink=0.5)

# Plot exact solution:
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, uex(X, Y), cmap=cm.coolwarm, linewidth=0)
fig.colorbar(surf, shrink=0.5)

# Error:
error = np.max(np.abs(uex(X,Y) - u))/np.max(np.abs(uex(X, Y)))
print('Error:', error)