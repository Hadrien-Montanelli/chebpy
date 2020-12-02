#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:37:10 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library import:
import numpy as np

# Chebpy imports:
from chebpy import chebpts, coeffs2vals, vals2coeffs

# %% Approximation (1D):    
f = lambda x: np.exp(np.sin(x)**2)
N = 100
x = chebpts(N)
error = coeffs2vals(vals2coeffs(f(x))) - f(x)
print('Error (approx):', np.max(np.abs(error)))

# %% Approximation (2D):
f = lambda x,y: np.exp(-10*(x + y)**2)
N = 100
x = chebpts(N)
y = chebpts(N)
X, Y = np.meshgrid(x, y)
values = f(X, Y)
coeffs = vals2coeffs(vals2coeffs(values).T).T
values2 = coeffs2vals(coeffs2vals(coeffs).T).T
error = values2 - values
print('Error (approx):', np.max(np.abs(error)))