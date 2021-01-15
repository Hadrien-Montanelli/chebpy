#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:29:37 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
from math import pi
import numpy as np

# Chebpy imports:
from chebpy.trig import trigpts, quadwts

# %% Integrate f(x) = 1 + sin(4*pi*x) on [-1,1].

# Function:
f = lambda x: 1 + np.sin(4*pi*x)

# Grid and weights:
n = 12
x = trigpts(n)
w = quadwts(n)

# Compute integral with quadrature:
I = w @ f(x)

# Compute error:
Iex = 2
error = np.abs(I - Iex)
print(f'Error: {error:.2e}')

# %% Integrate f(x) = 1 + sin(4*x) on [-pi,pi].

# Function:
f = lambda x: 1 + np.sin(4*x)

# Grid and weights:
n = 12
x = trigpts(n, [-pi, pi])
w = quadwts(n, [-pi, pi])

# Compute integral with quadrature:
I = w @ f(x)

# Compute error:
Iex = 2*pi
error = np.abs(I - Iex)
print(f'Error: {error:.2e}')