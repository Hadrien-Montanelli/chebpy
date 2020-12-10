#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:59:37 2020

Copyright 2020 by Hadrien Montanelli.
"""
# %% Imports.

# Standard library imports:
import numpy as np
from scipy.sparse import csr_matrix

# Chebpy imports:
from chebpy.cheb import sphankel

# %% Test 1.

col = np.array([1, 2, 3, 4])
H = sphankel(col)
print(csr_matrix.todense(H))