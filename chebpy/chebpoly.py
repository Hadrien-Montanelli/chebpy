#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:26:59 2020

Copyright 2020 by Hadrien Montanelli.

"""
def chebpoly(n):
    if (n==0):
        T = lambda x: 1 + 0*x
    elif (n==1):
        T = lambda x: x
    else:
        T1 = chebpoly(n-1)
        T2 = chebpoly(n-2)
        T = lambda x: 2*x*T1(x) - T2(x)
    return T