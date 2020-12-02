#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:28:05 2020

@author: montanelli
"""
# Standard library imports:
import numpy as np

def vals2coeffs(values):
    n = len(values)
    tmp = np.concatenate((values[-1:0:-1], values[0:n-1]))
    coeffs = np.fft.ifft(tmp)
    coeffs = np.real(coeffs[0:n])
    coeffs[1:n-2] = 2*coeffs[1:n-2]
    return coeffs