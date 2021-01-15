#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:27:31 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

def quadwts(n, dom=[-1, 1]):
    """Return n weights for trapezoidal rule."""
    w = 2/n*np.ones(n)
    w = (dom[1] - dom[0])/2*w
    return w