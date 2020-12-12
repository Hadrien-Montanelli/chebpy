#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:28:05 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

def vals2coeffs(values):
    """Convert values at equispaced points to Fourier coefficients."""
    n = len(values)
    coeffs = np.fft.fft(1/n*values, axis=0)
    if (len(coeffs.shape) > 1):
        coeffs = np.fft.fftshift(coeffs, axes=1)
    else:
        coeffs = np.fft.fftshift(coeffs)
    coeffs = (-1)**np.arange(-n/2, n/2) * coeffs
    return coeffs