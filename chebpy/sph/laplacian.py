#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:40:21 2020

@author: montanelli
"""
# Standard library import:
from math import pi
import numpy as np
from scipy.sparse import block_diag, eye, lil_matrix
from scipy.sparse.linalg import inv

# Chebpy imports:
from ..trig.diffmat import diffmat
from ..trig.multmat import multmat

def laplacian(n, flag_mean=1, flag_sin2=1):
    """Return the (n*n) x (n*n) Laplacian on the sphere."""
    
    # Differentiation and multiplication matrices:
    dom = [-pi, pi]
    I = eye(n)
    D1 = diffmat(n, 1, dom)
    D2 = diffmat(n, 2, dom)
    Tsin2 = multmat(n, lambda x: np.sin(x)**2, dom)
    Tcossin = multmat(n, lambda x: np.cos(x)*np.sin(x), dom)
    
    # Blocks (not multiplied by sin^2):
    if (flag_sin2 == 0):
        Tsin2inv = inv(Tsin2)
        blocks = []
        for k in range(n):
            block = D2 + Tsin2inv @ Tcossin @ D1 + D2[k, k]*Tsin2inv
            if (k == int(n/2) and flag_mean == 1): # mean condition
                mm = np.arange(-n/2, n/2)
                mm[int(n/2)-1] = 0
                mm[int(n/2)+1] = 0
                en = 2*pi*(1 + np.exp(1j*pi*mm))/(1 - mm**2)
                en[int(n/2)-1] = 0
                en[int(n/2)+1] = 0
                block = lil_matrix(block)
                block[int(n/2), :] = en
            blocks.append(block)
            
    # Blocks (multiplied by sin^2):
    if (flag_sin2 == 1):
        blocks = []
        for k in range(n):
            block = Tsin2 @ D2 + Tcossin @ D1 + D2[k, k]*I
            if (k == int(n/2) and flag_mean == 1): # mean condition
                mm = np.arange(-n/2, n/2)
                mm[int(n/2)-1] = 0
                mm[int(n/2)+1] = 0
                en = 2*pi*(1 + np.exp(1j*pi*mm))/(1 - mm**2)
                en[int(n/2)-1] = 0
                en[int(n/2)+1] = 0
                block = lil_matrix(block)
                block[int(n/2), :] = en
            blocks.append(block)
    
    # Assemble Laplacian:
    L = block_diag(blocks, format='csr')
    
    return L