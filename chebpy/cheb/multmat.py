#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:39:54 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard imports:
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import eye
from scipy.sparse import hstack
from scipy.sparse import vstack
from scipy.sparse import spdiags

# Chebpy imports:
from .chebpts import chebpts
from .spconvert import spconvert
from .sphankel import sphankel
from .sptoeplitz import sptoeplitz
from .vals2coeffs import vals2coeffs

def multmat(n, f, dom=[-1, 1], lam=0):
    """Return the n x n multiplication by f matrix in the C^{(lam)} basis."""
    # Get the Chebyshev coefficients:
    x = chebpts(n, dom)
    F = vals2coeffs(f(x))
        
    # Multiplication in the Chebyshev bsis:
    if (lam == 0):
        
        # Scale first term:
        digits = int(15 - np.floor(np.log10(np.max(np.abs(F)))))
        F = np.round(F, digits)
        F[0] = 2*F[0]
        
        # Toeplitz part:
        T = sptoeplitz(F, F)
        
        # Hankel part:
        H = sphankel(F[1:])
        H = hstack([H, csr_matrix(np.zeros([n-1, 1]))])
        H = vstack([csr_matrix(np.zeros([1, n])), H])
        
        # Assemble M:
        M = 1/2*(T + H)

    # Multiplication in the C^{(lam)} basis:
    else:
        
        # Convert coefficients to the C^{(lam)} basis:
        for k in range(lam):
            S = spconvert(n, k)
            F = S @ F
        
        # Assemble matrices:
        M0 = eye(n)
        d0 = np.ones(1)
        d1 = np.concatenate((d0, np.arange(2*lam, 2*lam + n - 1)), axis=0)
        d1 = d1/np.concatenate((d0, 2*np.arange(lam + 1, lam + n)), axis=0)
        d2 = np.arange(1, n + 1)/(2*np.arange(lam, lam + n))
        diags = np.zeros([3, n])
        diags[0, :] = d2
        diags[2, :] = d1
        Mx = spdiags(diags, [-1, 0, 1], n, n)
        M1 = 2*lam*Mx

        # Assemble M with three-term recurrence: O(n^2) complexity.
        M = F[0]*M0 + F[1]*M1
        for k in range(1, n-1):
            M2 = 2*(k + lam)/(k + 1)*(Mx @ M1) - (k + 2*lam - 1)/(k + 1)*M0
            M = M + F[k+1]*M2
            M0 = M1
            M1 = M2
            if (np.abs(F[k+2:]).all() < np.finfo(float).eps):
                break
        M = np.round(M, 15)  
            
    return M