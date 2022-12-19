# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 09:44:09 2022

"""

import numpy as np

# %%
class funct:
    def __init__(self, f):
        #if type(f) != 'str':
            #raise ValueError('f should be string')
        self.f = f
    def __call__(self, y):
        if self.f == 'f1':
            return 1/np.tan(y[0])  #cot als 1/tan
        if self.f == 'f2':
            return np.sin(y[0]) + np.cos(y[1])
        if self.f == 'f3':
            res = -np.shape(y)**2
            for i in range(np.shape(y)):
                res += np.exp(-(y[i]**2))
            return res
        raise ValueError('wrong function')
        
    
    def derivative(self, y):
        if self.f == 'f1':
            return np.array([[-1/((np.sin(float(y[0])**2)))]])
        if self.f == 'f2':
            return np.array([[np.cos(y[0]),-np.sin(y[1])]])
        if self.f == 'f3':
            res = np.zeros(np.shape(y))
            for i in range(len(y)):
                res[i] = np.exp(-(y[i]**2))*(-2*y[i])
            return res
        raise ValueError('wrong function')


def newton_like(variant, f, x_0, dfz=None, x_1=None, tol=1e-8, max_iter=50):
    '''Abbruch: |xk-x(k-1)| < tol oder max_iter ist erreicht
    - assume x_0 is np.array'''
    if(type(f)==None or type(x_0)==None):
        raise ValueError('f or x_0 is given incorrectly!')
    
    x_k = x_0.copy()
    f = funct(f)
    
    if variant=='standard':
        for i in range (max_iter):
            x_k1 = x_k - np.linalg.pinv(f.derivative(x_k)) * f.__call__(x_k)  #TODO: test errors... works, for f1, error for function f2, f3, but I don't know how to fix it
            #works with Moore-Penrose-Inverse, because Matrix not always square
            if np.linalg.norm(x_k-x_k1) < tol:
                return x_k1
            x_k = x_k1.copy()
        return x_k
    
    elif variant=='secant':
        if x_1==None:
            raise ValueError('In Variant==secant: x_1 must be given!')
        x_k1 = x_1.copy()
        if x_0.shape[0]==1:
            # 1 D Fall implementieren
            for i in range (max_iter):
                f_sup = f.__call__(x_k1)-f.__call__(x_k)
                if f_sup == 0:
                    return x_k1 
                x_k2 = x_k1 - ((x_k1 - x_k)/(f_sup))*f.__call__(x_k1)
                if np.linalg.norm(x_k1-x_k2) < tol:
                    return x_k2
                x_k = x_k1.copy()
                x_k1 = x_k2.copy()
            return x_k1
        # dim > 1
        for i in range (max_iter):
            x_k2 = np.linalg.solve(f.derivative(x_k1), (f.derivative(x_k1)*x_k1 - f.__call__(x_k1)))
            #x_k2 = x_k1 - (np.linalg.inv(f.derivative(x_k1))@f.__call__(x_k1))
            if np.linalg.norm(x_k1-x_k2) < tol:
                return x_k2
            x_k = x_k1.copy()
            x_k1 = x_k2.copy()
        return x_k1
    
    elif variant=='simple':
        if dfz==None:
            raise ValueError('In variant== simple: dfz is not given!')
        dfz1 = np.linalg.inv(dfz)
        for i in range (max_iter):
            x_k1 = x_k - (dfz1@f.__call__(x_k))
            if np.linalg.norm(x_k-x_k1) < tol:
                return x_k1
            x_k = x_k1.copy()
        return x_k
    
    else:
        raise ValueError('variant is not known!')
        
print(newton_like('standard', 'f1', np.array([1])))
print(newton_like('standard', 'f2', np.array([0,1])))
print(newton_like('standard', 'f3', np.ones(37)))
#Errors for f2, f3, but don't know how to fix them

print(newton_like('secant', 'f1', np.array([1])))
print(newton_like('secant', 'f2', np.array([0,1])))
print(newton_like('secant', 'f3', np.ones(37)))
#Errors for variant 'secant', because currently no x1 given
#maybe also other errors, not yet tested

print(newton_like('simple', 'f1', np.array([1])))
print(newton_like('simple', 'f2', np.array([0,1])))
print(newton_like('simple', 'f3', np.ones(37)))
#errors for variant 'simple', because currently no matrix dfz given
#also other error, but haven't fixed them til now