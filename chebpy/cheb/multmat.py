#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:39:54 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard imports:
import numpy as np
from scipy.linalg import hankel
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags

# Chebpy imports:
from .chebpts import chebpts
from .spconvert import spconvert
from .vals2coeffs import vals2coeffs

def multmat(n, f, dom=[-1, 1], lam=0):
    """Return the n x n multiplication by f matrix in the C^{(lam)} basis."""
    # Get the Chebyshev coefficients:
    x = chebpts(n, dom)
    F = vals2coeffs(f(x))
    
    # Multiplication in the Chebyshev bsis:
    if (lam==0):
        
        # Scale first term:
        F[0] = 2*F[0]
        
        # Toeplitz part:
        T = toeplitz(F)
        
        # Hankel part:
        H = hankel(F[1:])
        H = np.concatenate((H, np.zeros([n-1, 1])), axis=1)
        H = np.concatenate((np.zeros([1, n]), H), axis=0)
        
        # Assemble M:
        M = csr_matrix(np.round(1/2*(T + H), 15))
    
    # Multiplication in the C^{(lam)} basis:
    else:
        
        # Convert coefficients to the C^{(lam)} basis:
        for k in range(lam):
            S = spconvert(n, k)
            F = S @ F
            
        # Assemble matrices:
        M0 = np.eye(n)
        d0 = np.ones(1)
        d1 = np.concatenate((d0, np.arange(2*lam, 2*lam + n - 1)), axis=0)
        d1 = d1/np.concatenate((d0, 2*np.arange(lam + 1, lam + n)), axis=0)
        d2 = np.arange(1, n + 1)/(2*np.arange(lam, lam + n))
        diags = np.zeros([3, n])
        diags[0, :] = d2
        diags[2, :] = d1
        Mx = spdiags(diags, [-1, 0, 1], n, n)
        M1 = 2*lam*Mx
        
        # Assemble M with three-term recurrence:
        M = F[0]*M0 + F[1]*M1
        for k in range(1, n-1):
            M2 = 2*(k + lam)/(k + 1)*(Mx @ M1) - (k + 2*lam - 1)/(k + 1)*M0
            M = M + F[k+1]*M2
            M0 = M1
            M1 = M2
            if (np.abs(F[k+2:]).all() < 1e-15):
                break
        M = csr_matrix(np.round(M, 15))    
            
    return M
