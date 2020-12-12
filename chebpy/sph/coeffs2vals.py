#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:28:26 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Chebpy imports:
from ..trig.coeffs2vals import coeffs2vals as c2v
    
def coeffs2vals(coeffs):
    """Convert Fourier coefficients to values at equispaced points (DFS)."""
    values = c2v(c2v(coeffs).T).T
    return values