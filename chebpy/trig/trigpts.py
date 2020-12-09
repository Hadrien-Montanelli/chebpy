#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:27:31 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library imports:
import numpy as np

def trigpts(n, dom=[-1, 1]):
    """Return n equispaced points."""
    x = np.linspace(-1, 1, n+1)
    x = x[:-1]
    return x