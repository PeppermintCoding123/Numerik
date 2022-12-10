# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 12:56:44 2022

@author: laura
"""

#%%
import numpy as np
import sympy as sym

print(sym.sin(3))
def Jacobian(v_str, f_list):
    vars = sym.symbols(v_str)
    f = sym.sympify(f_list)
    J = sym.zeros(len(f),len(vars))
    for i, fi in enumerate(f):
        for j, s in enumerate(vars):
            J[i,j] = sym.diff(fi, s)
    return J

J = Jacobian('u1 u2', ['sym.sin(u1)+sym.sin(u2)','u1**2'])

print(J)
v_str = 'u1 u2'
v_str = list(v_str)

u1 = 5
u2 = 3
x = (5,3)
#vars = sym.symbols(v_str)
l = len(J)
a = J[0]
b = J[1]
c = J[2]
d = J[3]
for i in range(len(J)):
    J[i] = [J[i].replace(v_str[k],x[k]) for k in range(len(v_str))]

print(J)