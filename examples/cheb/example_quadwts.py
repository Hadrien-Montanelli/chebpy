#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:29:37 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
import numpy as np

# Chebpy imports:
from chebpy.cheb import chebpts, quadwts

# %% Integrate f(x) = exp(x) on [-1,1].

# Function:
f = lambda x: np.exp(x)

# Grid and weights:
n = 12
x = chebpts(n)
w = quadwts(n)

# Compute integral with quadrature:
I = w @ f(x)

# Compute error:
Iex = np.exp(1) - np.exp(-1)
error = np.abs(I - Iex)
print(f'Error: {error:.2e}')

# %% Integrate f(x) = exp(x) on [0,2].

# Function:
f = lambda x: np.exp(x)

# Grid and weights:
n = 12
x = chebpts(n, [0, 2])
w = quadwts(n, [0, 2])

# Compute integral with quadrature:
I = w @ f(x)

# Compute error:
Iex = np.exp(2) - np.exp(0)
error = np.abs(I - Iex)
print(f'Error: {error:.2e}')