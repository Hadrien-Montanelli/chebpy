#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:56:16 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
from math import pi
import numpy as np

# Chebpy imports:
from chebpy.trig import feval, trigpts, vals2coeffs

# %% Evaluate f(x) = cos(pi*x)*exp(-sin(pi*x)^2).

# Function:
f = lambda x: np.cos(pi*x)*np.exp(-np.sin(pi*x)**2)

# Equispaced grid:
n = 100
x = trigpts(n)
F = vals2coeffs(f(x))

# Evaluation grid:
xx = np.linspace(-1, 1, 100)
vals = feval(F, xx)

# Error:
error = np.max(np.abs(vals - f(xx)))/np.max(np.abs(f(xx)))
print(f'Error: {error:.2e}')