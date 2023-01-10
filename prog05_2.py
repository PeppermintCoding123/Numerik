# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:06:12 2023

@author: laura
"""
import numpy as np
#aus Aufgabe 1
def Horner_polyval(x, a):
    m,n = np.shape(a)
    if m == 0:
        return 0
    P = a[n-1]*x + a[n-2]
    for i in range(2,n):
        P *= x+ a[n-1-i]
    return P

#%%Aufgabe 2

class Vandermonde_model:
    def fit(self, x, y):
        self.x = x
        self.y = y
        if np.shape(self.x) != np.shape(self.y):
            raise ValueError('shape of x and y should be the same')
        m,n = np.shape(self.x)
        self.p = np.linalg.solve(np.vander(self.x,n,increasing=True), self.y)
    
    def __call__(self, x):
        return Horner_polyval(x, self.p)
    
    def add_points(self, x, y):
        pass#TODO im Tutorium nachfragen, wie die Aufgabenstellung zu verstehen ist

class Lagrange_model:
    def fit(self, x, y):
        pass
    
    def __call__(self, x):
        pass
    
    def add_points(self, x, y):
        pass

class Newton_model:
    def fit(self, x, y):
        pass
    
    def __call__(self, x):
        pass
    
    def add_points(self, x, y):
        pass
    