#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:56:16 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library import:
import numpy as np

# Chebpy imports:
from chebpy.cheb import chebpts, feval, vals2coeffs

# %% Evaluate f(x) = cos(x)*exp(-x^2).

# Function:
f = lambda x: np.cos(x)*np.exp(-x**2)

# Chebyshev grid:
n = 30
x = chebpts(n)
F = vals2coeffs(f(x))

# Evaluation grid:
xx = np.linspace(-1, 1, 100)
vals = feval(F, xx)

# Error:
error = np.max(np.abs(vals - f(xx)))/np.max(np.abs(f(xx)))
print('Error:', error)