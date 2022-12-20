# -*- coding: utf-8 -*-

import numpy as np

# %% 

class f1:
    def __call__(self, y):
        return 1/np.tan(y[0])
    def derivative(self, y):
        return np.array([[-1/((np.sin(float(y[0])**2)))]])

#TODO:@Laura: können wir die y arrays eine stelle weiter rausholen als die Teife wo es gerade ist? Wenn das dich verwirrt, einfach ignorieren.
class f2:
    def __call__(y):
        #return np.sin(y[0]) + np.cos(y[1])
        #ist das nicht die alte Version, müsste das nicht eigentlich Folgendes sein:
        return np.array([np.sin(y[0]),np.cos(y[1])])
    def derivative(y):
        return np.array([[np.cos(y[0][0]),0],[0,-np.sin(y[1][0])]])

class f3:
    def __call__(y):
        tmp = np.zeros(np.shape(y)[0])
        for i in range(np.shape(y)[0]):
            tmp = np.exp(-(y[i]**2)) -1
        return tmp
    def derivative(y):
        tmp = np.zeros(np.shape(y)[0], np.shape(y)[0])
        for i in range(np.shape(y)[0]):
            tmp[i][i] = np.exp(-(y[i]**2))*(-2*y[i])
        return tmp

# %%
def newton_like(f, x_0, dfz=None, x_1=None, tol=1e-8, max_iter=50, delta=0.01, variant='standard'):
    '''Abbruch: |xk-x(k-1)| < tol oder max_iter ist erreicht
    - assume x_0 is np.array'''
    if(f is None or x_0 is None): # ---Macht es einen Unterschied ob man das so oder mit f is Nonen Überprüft?
        raise ValueError('f or x_0 is given incorrectly!')
    
    x_k = x_0.copy()
    n = np.shape(x_0)[0]
    if variant=='standard':
        for i in range (max_iter):
            x_k1 = x_k - np.linalg.pinv(f.derivative(x_k)) * f.__call__(x_k) 
            # sollen wir überhaupt np.linalg.pinv aufrufen?
            # soll man hier More-Penroseinverse benutzen? Wei erstellt man die?+
            '''
            A =f.derivative(x_k) 
            n,m = Np.shape(A)
            if
            
            '''
            print('---Das ist x_k1---')
            print(x_k1)
            #test errors... works, for f1, error for function f2, f3, but I don't know how to fix it
            #works with Moore-Penrose-Inverse, because Matrix not always square
            if np.linalg.norm(x_k-x_k1) < tol:
                return x_k1
            x_k = x_k1.copy()
        return x_k
    
    elif variant=='secant':
        if isinstance(f, f1):
            if x_1 is None:
                raise ValueError('In Variant==secant: x_1 must be given!')
            x_k1 = x_1.copy()
            pass
        else:
            for k in range(max_iter):
                #Jk erstellen
                Jk = np.zeros((n, n))
                e = np.zeros(n)
                for j in range(n):
                    e[j] = 1
                    for i in range(n):
                        Jk[i] = (1/delta) * (f.__call__(y = (x_k + (delta*e))) - f.__call__(x_k))
                #x_k+1 erstellen  (1/delta) * (f2.__call__(y = (x_k + (delta*e))) - f2.__call__(x_k))
                x_k1 = x_k - np.linalg.inv(Jk)@f.__call__(x_k)
                x_k = x_k1
            return x_k
        
            '''# 1 D Fall implementieren
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
            #TODO: statt f.derivative estwas anderes, oder nicht
            #x_k2 = x_k1 - (np.linalg.inv(f.derivative(x_k1))@f.__call__(x_k1))
            if np.linalg.norm(x_k1-x_k2) < tol:
                return x_k2
            x_k = x_k1.copy()
            x_k1 = x_k2.copy()
        return x_k1'''
    
    elif variant=='simple':
        if dfz is None:
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
   
# %%
'''print(newton_like('standard', 'f1', np.array([1])))
print(newton_like('standard', 'f2', np.array([[0],[1]])))
print(newton_like('standard', 'f3', np.ones(37)))
#Errors for f2, f3, but don't know how to fix them'''

#print(newton_like('secant', f1, np.array([1])))
#print(newton_like(f2, np.array([0,1]), variant='secant'))
print(newton_like(f3, np.ones(20), variant='secant'))
#Errors for variant 'secant', because currently no x1 given
#maybe also other errors, not yet tested

'''print(newton_like('simple', 'f1', np.array([1])))
print(newton_like('simple', 'f2', np.array([0,1])))
print(newton_like('simple', 'f3', np.ones(37)))
#errors for variant 'simple', because currently no matrix dfz given
#also other error, but haven't fixed them til now'''