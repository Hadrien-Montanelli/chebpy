#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:37:10 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
from math import pi
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

# Chebpy imports:
from chebpy.trig import trigpts
from chebpy.sph import vals2coeffs, feval, mean, sph_harm

# %% Test.

# Grid points:
n = 32
dom = [-pi, pi]
lam = trigpts(n, dom)
tt = trigpts(n, dom)
LAM, TT = np.meshgrid(lam, tt)

# Spherical harmonic:
l = 4 # has to be >= 0
m = -3 # -l <= m <= +l
Y = sph_harm(l, m)
    
# Plot:
plt.figure()
plt.contourf(LAM, TT, feval(Y, LAM, TT), 40, cmap=cm.coolwarm)
plt.colorbar()

# Check orthonormality:
F = vals2coeffs(feval(Y, LAM, TT) * feval(Y, LAM, TT))
mu = mean(F)
print(f'Mean of Y_{l,m}^2: {mu:.2e}')