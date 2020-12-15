#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:51:33 2020

@author: montanelli
"""
# Standard imports:
import numpy as np
from scipy.linalg import toeplitz

def fbmc(F):
    """Enforce the BMC-I symmetry conditions for the DFS coefficients F."""
    
    # Get the dimension:
    n = len(F)
    Fbmc = np.zeros([n, n], dtype=complex)
    
    # %% Step 1: enforce f_{j,k} = -f_{-j,k} for odd k.   
    
    # Exctract odd modes in k and all modes in j:
    idx_k = 2*np.arange(int(n/2)) + 1
    idx_j = np.arange(n)
    idx_k, idx_j = np.meshgrid(idx_k, idx_j)
    Fodd = F[idx_j, idx_k]
    
    # Matrices:
    I = np.eye(int(n/2)+1, n)
    col = np.zeros(int(n/2)+1)
    col[1] = 1
    row = np.zeros(n)
    J = toeplitz(col, row)
    A = I + np.fliplr(J)
    A[-1, int(n/2)] = 1
    
    # Minimum Frobenius-norm solution:
    C = A.T @ np.linalg.inv(A @ A.T) @ (A @ Fodd)
    Fbmc[idx_j, idx_k] = F[idx_j, idx_k] - C
    
    # %% Step 2: enforce f_{j,k} = f_{-j,k} for even k.   
    
    # Exctract even modes in k and all modes in j (but exclude j=-n/2, 0):
    idx_k = 2*np.arange(int(n/2))
    idx_j = np.arange(1, int(n/2))
    idx_j = np.concatenate((idx_j, np.arange(int(n/2) + 1, n)))
    idx_k, idx_j = np.meshgrid(idx_k, idx_j)
    Feven = F[idx_j, idx_k]
    
    # Matrices:
    I = np.eye(int(n/2)-1, n-2)
    A = I - np.fliplr(I)
    
    # Minimum Frobenius-norm solution:
    C = A.T @ np.linalg.inv(A @ A.T) @ (A @ Feven)
    Fbmc[idx_j, idx_k] = F[idx_j, idx_k] - C
    
    # %% Step 3: enforce Re(f_{j,k}) = -Re(f_{j,-k}) for odd k.   
    
    # Exctract odd modes in k and all modes in j:
    idx_k = 2*np.arange(int(n/2)) + 1
    idx_j = np.arange(n)
    idx_k, idx_j = np.meshgrid(idx_k, idx_j)
    Fodd = np.real(Fbmc[idx_j, idx_k])
    
    # Matrices:
    I = np.eye(int(n/4), int(n/2))
    B = I + np.fliplr(I)

    # Minimum Frobenius-norm solution:
    C = (Fodd @ B.T) @ np.linalg.inv(B @ B.T) @ B
    Fbmc[idx_j, idx_k] = Fbmc[idx_j, idx_k] - C
    
    # %% Step 4: enforce Re(f_{j,k}) = Re(f_{j,-k}) for even k.
    
    # Exctract even modes in k (but exclude k=-n/2, 0) and all modes in j:
    idx_k = 2*np.arange(1, int(n/4))
    idx_k = np.concatenate((idx_k, 2*np.arange(int(n/4)+1, int(n/2))))
    idx_j = np.arange(n)
    idx_k, idx_j = np.meshgrid(idx_k, idx_j)
    Feven = np.real(Fbmc[idx_j, idx_k])
    
    # Matrices:
    I = np.eye(int(n/4)-1, int(n/2)-2)
    B = I - np.fliplr(I)
    
    # Minimum Frobenius-norm solution:
    C = (Feven @ B.T) @ np.linalg.inv(B @ B.T) @ B
    Fbmc[idx_j, idx_k] = Fbmc[idx_j, idx_k] - C
    
    # %% Step 5: enforce Im(f_{j,k}) = Im(f_{j,-k}) for odd k. 
    
    # Exctract odd modes in k and all modes in j:
    idx_k = 2*np.arange(int(n/2)) + 1
    idx_j = np.arange(n)
    idx_k, idx_j = np.meshgrid(idx_k, idx_j)
    Fodd = np.imag(Fbmc[idx_j, idx_k])
    
    # Matrices:
    I = np.eye(int(n/4), int(n/2))
    B = I - np.fliplr(I)

    # Minimum Frobenius-norm solution:
    C = (Fodd @ B.T) @ np.linalg.inv(B @ B.T) @ B
    Fbmc[idx_j, idx_k] = Fbmc[idx_j, idx_k] - 1j*C
    
    # %% Step 6: enforce Im(f_{j,k}) = -Im(f_{j,-k}) for even k. 
    
    # Exctract even modes in k (but exclude k=-n/2, 0) and all modes in j:
    idx_k = 2*np.arange(1, int(n/4))
    idx_k = np.concatenate((idx_k, 2*np.arange(int(n/4)+1, int(n/2))))
    idx_j = np.arange(n)
    idx_k, idx_j = np.meshgrid(idx_k, idx_j)
    Feven = np.imag(Fbmc[idx_j, idx_k])
    
    # Matrices:
    I = np.eye(int(n/4)-1, int(n/2)-2)
    B = I + np.fliplr(I)
    
    # Minimum Frobenius-norm solution:
    C = (Feven @ B.T) @ np.linalg.inv(B @ B.T) @ B
    Fbmc[idx_j, idx_k] = Fbmc[idx_j, idx_k] - 1j*C
    
    return Fbmc