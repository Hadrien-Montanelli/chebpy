#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:27:31 2020

@author: montanelli
"""
# Standard library imports:
import numpy as np

def chebpts(n):
    x = -np.cos(np.array([j for j in range(n)])*np.pi/(n-1))
    return x