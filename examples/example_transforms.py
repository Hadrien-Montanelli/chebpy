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

# %% Transforms (1D) on [-1,1].

f = lambda x: np.exp(-10*x**2)
n = 100
x = chebpts(n)
error = coeffs2vals(vals2coeffs(f(x))) - f(x)
print('Error (1D):', np.max(np.abs(error)))

# %% Transforms (1D) on [0,2].

f = lambda x: np.exp(-10*x**2)
n = 100
x = chebpts(n, [0, 2])
error = coeffs2vals(vals2coeffs(f(x))) - f(x)
print('Error (1D):', np.max(np.abs(error)))

# %% Transforms (2D) on [-1,1]x[-1,1].

f = lambda x,y: np.exp(-10*(x**2 + y**2))
n = 100
x = chebpts(n)
y = chebpts(n)
X, Y = np.meshgrid(x, y)
values = f(X, Y)
coeffs = vals2coeffs(vals2coeffs(values).T).T
values2 = coeffs2vals(coeffs2vals(coeffs).T).T
error = values2 - values
print('Error (2D):', np.max(np.abs(error)))

# %% Transforms (2D) on [0,2]x[-1,0].

f = lambda x,y: np.exp(-10*(x**2 + y**2))
n = 100
x = chebpts(n, [0, 2])
y = chebpts(n, [-1, 0])
X, Y = np.meshgrid(x, y)
values = f(X, Y)
coeffs = vals2coeffs(vals2coeffs(values).T).T
values2 = coeffs2vals(coeffs2vals(coeffs).T).T
error = values2 - values
print('Error (2D):', np.max(np.abs(error)))