#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:28:05 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Chebpy imports:
from ..trig.vals2coeffs import vals2coeffs as v2c
    
def vals2coeffs(values):
    """Convert values at equispaced points to Fourier coefficients (DFS)."""
    coeffs = v2c(v2c(values).T).T 
    return coeffs