#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:23:46 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library import:
import matplotlib.pylab as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Chebpy imports:
from chebpy import chebpoly, chebpts, coeffs2vals, vals2coeffs
from chebpy import diffmat, spconvert

# %% Solve u_xx + u_yy = f(x,y) on [-1,1]x[-1,1] with homogeneous Dirichlet 
# boundary conditions.