#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:28:26 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

def coeffs2vals(coeffs):
    """Convert Fourier coefficients to values at equispaced points."""
    n = len(coeffs)
    coeffs = (-1)**np.arange(-n/2, n/2) * coeffs
    if (len(coeffs.shape) > 1):
        coeffs = np.fft.ifftshift(coeffs, axes=1)
    else:
        coeffs = np.fft.ifftshift(coeffs)
    values = n*np.fft.ifft(coeffs, axis=0)
    values = np.real(values)
    return values