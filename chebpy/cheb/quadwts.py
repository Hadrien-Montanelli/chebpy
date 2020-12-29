#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:27:31 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

def quadwts(n, dom=[-1, 1]):
    """Return n weights for Clenshaw-Curtis quadrature."""
    c = 2/np.concatenate(([1],  1 - np.arange(2, n, 2)**2), axis=0)
    c = np.concatenate((c, c[int(n/2)-1:0:-1]), axis=0)
    w = np.real(np.fft.ifft(c))
    w[0] = w[0]/2
    w = np.concatenate((w, [w[0]]), axis=0)
    return w