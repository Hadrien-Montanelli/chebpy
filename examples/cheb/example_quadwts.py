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
n = 20
x = chebpts(n)
w = quadwts(n)

# Compute integral with quadrature:
I = np.sum(w*f(x))

# Compute error:
Iex = np.exp(1) - np.exp(-1)
error = np.abs(I - Iex)
print(f'Error: {error:.2e}')