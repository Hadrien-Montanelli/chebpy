#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 11:04:28 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library import:
from math import pi
import numpy as np
from scipy.sparse import block_diag
from scipy.sparse import csc_matrix
from scipy.sparse import eye
from scipy.sparse import kron
from scipy.sparse.linalg import splu

# Chebpy imports:
from chebpy.trig import diffmat, multmat

def start_imex(N, m, n, h, v0, alpha, q):
    """Start an IMEX scheme with q-1 steps of LIRK4."""
    
    # Construct Laplacian matrix (multiplied by Tsin2 and alpha):
    dom = [-pi, pi]
    I = eye(n)
    D1 = diffmat(n, 1, dom)
    D2 = diffmat(n, 2, dom)
    Tsin2 = multmat(n, lambda x: np.sin(x)**2, dom)
    Tcossin = multmat(n, lambda x: np.cos(x)*np.sin(x), dom)
    blocks = []
    for k in range(n):
        block = Tsin2 @ D2 + Tcossin @ D1 + D2[k, k]*I
        blocks.append(block)
    Lap = alpha*block_diag(blocks, format='csc')
    
    # Compute LU factorizations of LIRK4 matrices:
    Tsin2 = csc_matrix(kron(I, Tsin2))
    lu = splu(Tsin2)
    lua = splu(Tsin2 - 1/4*h*Lap)
    
    # Time-stepping loop:
    U = [v0, v0, v0, v0]
    NU = [N(v0), N(v0), N(v0), N(v0)]
    v = v0
    for i in range(1, q):
        Nv = N(v)
        w = Tsin2 @ v
        wa = w + h*Tsin2 @ (1/4*Nv)
        a = lua.solve(wa)
        Na = N(a)
        wb = w + h*Lap @ (1/2*a) + h*Tsin2 @ (-1/4*Nv + Na)
        b = lua.solve(wb)
        Nb = N(b)
        wc = w + h*Lap @ (17/50*a - 1/25*b)
        wc += h*Tsin2 @ (-13/100*Nv + 43/75*Na + 8/75*Nb)
        c = lua.solve(wc)
        Nc = N(c)
        wd = w + h*Lap @ (371/1360*a - 137/2720*b + 15/544*c)
        w += h*Tsin2 @ (-6/85*Nv + 42/85*Na + 179/1360*Nb - 15/272*Nc)
        d = lua.solve(wd)
        Nd = N(d)
        we = w + h*Lap @ (25/24*a - 49/48*b + 125/16*c - 85/12*d)
        we += h*Tsin2*(79/24*Na - 5/8*Nb + 25/2*Nc - 85/6*Nd)
        e = lua.solve(we)
        Ne = N(e)
        v += h*lu.solve(Lap @ (25/24*a - 49/48*b + 125/16*c - 85/12*d + 1/4*e))
        v += h*(25/24*Na - 49/48*Nb + 125/16*Nc - 85/12*Nd + 1/4*Ne)
        U[-1-i] = v
        NU[-1-i] = N(v)
        
    return U, NU