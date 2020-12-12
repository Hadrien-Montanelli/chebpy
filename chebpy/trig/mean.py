#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:28:26 2020

Copyright 2020 by Hadrien Montanelli.
"""
def mean(F):
    """Return the mean of the Fourier series with coefficients F."""
    n = len(F)
    mu = F[int(n/2)]
    return mu