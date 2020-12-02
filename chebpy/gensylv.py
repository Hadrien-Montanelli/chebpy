#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:00:03 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard library import:
import numpy as np

# Chebpy imports:
from .sylv import sylv

def gensylv(A1, B1, A2, B2, C):
    A2inv = np.linalg.inv(A2)
    A = A2inv @ A1
    B1inv = np.linalg.inv(B1)
    B = B2 @ B1inv
    return sylv(A, B, A2inv @ C @ B1inv)