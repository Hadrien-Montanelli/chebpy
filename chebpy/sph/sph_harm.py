#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:28:26 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard imports:
from math import factorial, pi, sqrt
import numpy as np
from scipy.special import lpmv

def sph_harm(l, m):
    """Return spherical harmonic Y_{l,m}."""
    if (m >=0):
        P = lambda x: lpmv(m, l, x)
        scl = sqrt((2 - (m==0))*(2*l + 1)*factorial(l - m)/factorial(l + m))
        Y = lambda ll, tt: scl/sqrt(4*pi) * P(np.cos(tt)) * np.cos(m*ll)
    else:
        m = abs(m)
        P = lambda x: lpmv(m, l, x)
        scl = sqrt((2 - (m==0))*(2*l + 1)*factorial(l - m)/factorial(l + m))
        Y = lambda ll, tt: scl/sqrt(4*pi) * P(np.cos(tt)) * np.sin(m*ll)
    return Y