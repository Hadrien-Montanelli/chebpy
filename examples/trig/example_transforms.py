#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:37:10 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library import:
from math import pi
import numpy as np

# Chebpy imports:
from chebpy.trig import trigpts, coeffs2vals, vals2coeffs

# %% Transforms (1D) on [-1,1].

f = lambda x: np.cos(10000*pi*x) + np.cos(5*pi*x)
n = 101
x = trigpts(n)
error = coeffs2vals(vals2coeffs(f(x))) - f(x)
print('Error (1D):', np.max(np.abs(error)))

# %% Transforms (1D) on [0,2*pi].

f = lambda x: np.exp(np.cos(10*pi*x)**2)
n = 120
x = trigpts(n, [0, 2*pi])
error = coeffs2vals(vals2coeffs(f(x))) - f(x)
print('Error (1D):', np.max(np.abs(error)))

# %% Transforms (2D) on [-1,1]x[-1,1].

f = lambda x, y:  np.exp(np.cos(10*pi*x)**2)*np.sin(pi*y)**2
n = 100
x = trigpts(n)
y = trigpts(n)
X, Y = np.meshgrid(x, y)
values = f(X, Y)
coeffs = vals2coeffs(vals2coeffs(values).T).T
values2 = coeffs2vals(coeffs2vals(coeffs).T).T
error = values2 - values
print('Error (2D):', np.max(np.abs(error)))

# %% Transforms (2D) on [0,2]x[-1,0].

f = lambda x, y:  np.exp(np.cos(10*pi*x)**2)*np.sin(pi*y)**2
n = 100
x = trigpts(n, [-pi, pi])
y = trigpts(n, [0, 4*pi])
X, Y = np.meshgrid(x, y)
values = f(X, Y)
coeffs = vals2coeffs(vals2coeffs(values).T).T
values2 = coeffs2vals(coeffs2vals(coeffs).T).T
error = values2 - values
print('Error (2D):', np.max(np.abs(error)))