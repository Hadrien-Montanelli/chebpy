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
from chebpy import chebpts, coeffs2vals, vals2coeffs
from chebpy import diffmat, spconvert

# %% Solve u_xx = f on [-1,1]x[-1,1] with Dirichlet boundary conditions.

# RHS:
f = lambda x,y: -(2*pi)**2*np.sin(2*pi*x)*np.sin(2*pi*y)

# Boundary condtions:
g1 = lambda y: 0*y # u(-1, y) = g1(y)
g2 = lambda y: 0*y # u(+1, y) = g2(y)
h1 = lambda x: 0*x # u(x, -1) = h1(x)
h2 = lambda x: 0*x # u(x, +1) = h2(x)

# Exact solution:
uex = lambda x,y: np.sin(2*pi*x)*np.sin(2*pi*y)

# Grid points:
N = 50
x = chebpts(N)
y = chebpts(N)
X, Y = np.meshgrid(x, y)

# Assemble differentiation matrices:
A1 = np.eye(N)
C1 = np.array(csr_matrix.todense(diffmat(N, 2)))

# Assemble boundary conditions:
Bx = np.zeros([2, N])
By = np.zeros([2, N])
G = np.zeros([2, N])
H = np.zeros([2, N])
for k in range(N):
    Bx[0, k] = (-1)**k
    By[0, k] = (-1)**k
Bx[1, :] = np.ones(N)
By[1, :] = np.ones(N)   
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

# # Assemble right-hand side:
F = vals2coeffs(vals2coeffs(f(X, Y)).T).T
S0 = spconvert(N, 0)
S1 = spconvert(N, 1)
F = F @ (S1 @ S0).T

# Assemble matrices for the linear system:
Ft = F - A1[:N, :2] @ H @ C1.T - (A1 - A1[:N, :2] @ By) @ G.T @ C1[:N, :2].T
A1t = A1 - A1[:N, :2] @ By
C1t = C1 - C1[:N, :2] @ Bx

# Solve the linear system:
A1t = A1t[:N-2, 2:] # A12
C1t = C1t[:N-2, 2:] # C21
Ft = Ft[:N-2, :N-2] # F11 
U22 = np.linalg.inv(A1t) @ Ft @ np.linalg.inv(C1t.T)

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

# Error:
error = np.max(np.abs(uex(X,Y) - u))/np.max(np.abs(uex(X,Y)))
print('Error:', error)