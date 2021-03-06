#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:27:31 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

def chebpts(n, dom=[-1, 1]):
    """Return n Chebyshev points of the second kind."""
    x = -np.cos(np.array([j for j in range(n)])*np.pi/(n-1))
    x = dom[1]*(x + 1)/2 + dom[0]*(1 - x)/2
    return x