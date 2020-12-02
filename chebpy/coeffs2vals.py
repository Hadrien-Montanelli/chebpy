#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:28:26 2020

@author: montanelli
"""
# Standard library imports:
import numpy as np

def coeffs2vals(coeffs):
    n = len(coeffs)
    coeffs[1:n-2] = 1/2*coeffs[1:n-2]
    tmp = np.concatenate((coeffs, coeffs[-2:0:-1]))
    values = np.fft.fft(tmp)
    values = np.real(values[n-1::-1])
    return values