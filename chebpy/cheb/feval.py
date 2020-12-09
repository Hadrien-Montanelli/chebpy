#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:52:50 2020

Copyright 2020 by Hadrien Montanelli.
"""
def feval(F, x):
    """Evaluate Chebyshev series with coefficients F at x."""
    b1 = 0
    b2 = 0
    n = len(F) - 1
    for k in range(n, 0, -1):
        b = F[k] + 2*x*b1 - b2
        b2 = b1
        b1 = b
    y = F[0] + x*b1 - b2
    return y