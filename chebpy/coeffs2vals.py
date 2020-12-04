#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:28:26 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

def coeffs2vals(coeffs):
    """Convert values at Chebyshev points to Chebyshev coefficients."""
    n = len(coeffs)
    coeffs2 = coeffs.copy()
    coeffs2[1:n-2] = 1/2*coeffs2[1:n-2]
    tmp = np.concatenate((coeffs2, coeffs2[-2:0:-1]))
    values = np.fft.fft(tmp, axis=0)
    values = np.real(values[n-1::-1])
    return values