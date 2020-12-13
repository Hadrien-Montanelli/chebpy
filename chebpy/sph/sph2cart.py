#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 10:15:44 2020

Copyright 2020 by Hadrien Montanelli.
"""
# Standard imports:
import numpy as np

def sph2cart(u, v):
    """Return the Cartesian components of vector field V=(u,v)^T."""
    Vx = lambda ll, tt: v(ll,tt)*np.cos(tt)*np.cos(ll) - u(ll,tt)*np.sin(ll)
    Vy = lambda ll, tt: v(ll,tt)*np.cos(tt)*np.sin(ll) + u(ll,tt)*np.cos(ll)
    Vz = lambda ll, tt: -v(ll,tt)*np.sin(tt)
    return Vx, Vy, Vz