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

# Chebpy imports:
from chebpy import chebpts, coeffs2vals, vals2coeffs
from chebpy import diffmat, gensylv

# %% Solve u_xx + u_yy = f(x,y) on [-1,1]x[-1,1] with 0 Dirichlet conditions.

# Right-hand side:
f = lambda x,y: np.exp(-100*(x + y)**2)

# Assemble differentiation matrices:
N = 3
x = chebpts(N)
y = chebpts(N)
X, Y = np.meshgrid(x, y)
Bx = np.zeros([2, N])
By = np.zeros([2, N])
H = np.zeros([2, N])
G = np.zeros([2, N])
for k in range(N):
    Bx[0, k] = (-1)**k
    Bx[1, k] = 1
    By[0, k] = (-1)**k
    By[1, k] = 1
A1 = np.eye(N)
C1 = np.array(csr_matrix.todense(diffmat(N, 2)))
A2 = np.array(csr_matrix.todense(diffmat(N, 2)))
C2 = np.eye(N)
Bx_hat = Bx[0:2, 0:2]
Bx = np.linalg.inv(Bx_hat) @ Bx
G = np.linalg.inv(Bx_hat) @ G
By_hat = By[0:2, 0:2]
By = np.linalg.inv(By_hat) @ By
H = np.linalg.inv(By_hat) @ H

# # Assemble right-hand side:
F = vals2coeffs(vals2coeffs(f(X, Y)).T).T

# Assemble matrices for the the generalized Sylvester equation:
Ft = F - A1[:N, :2] @ H @ C1.T - (A1 - A1[:N, :2] @ By) @ G.T @ C1[:N, :2].T
Ft = Ft - A2[:N, :2] @ H @ C2.T - (A2 - A2[:N, :2] @ By) @ G.T @ C2[:N, :2].T
A1t = A1 - A1[:N, :2] @ By
A2t = A2 - A2[:N, :2] @ By
C1t = C1 - C1[:N, :2] @ Bx
C2t = C2 - C2[:N, :2] @ Bx
print(A1t)
print(A2t)
print(C1t.T)
print(C2t.T)

# Solve the generalized Sylvester equation and assemble solution:
# U22 = gensylv(A1t, C1t.T, A2t, C2t.T, Ft)
# U22 = U22[2:, 2:]
# U12 = H[:, 2:] - By[:, 2:] @ U22
# U21 = G[:, 2:].T - U22 @ Bx[:, 2:].T
# U11 = H[:, 0:2] - By[:, 2:] @ U21
# U1 = np.concatenate((U11, U12), axis=1)
# U2 = np.concatenate((U21, U22), axis=1)
# U = np.concatenate((U1, U2), axis=0)

# Plot solution:
# u = coeffs2vals(coeffs2vals(U).T).T
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, u, cmap=cm.coolwarm, linewidth=0)
# fig.colorbar(surf, shrink=0.5)