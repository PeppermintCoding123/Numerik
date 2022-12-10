import numpy as np
import sympy as sym

# %%
class funct:
    def __init__(self, v_str, f_list):
        vars = sym.symbols(v_str)
        self.f = sym.sympify(f_list)
    def __call__(self, x):
        '''f_x = sym.zeros(len(self.f))
        for i, fi in enumerate(self.f):'''
            # do something...
        pass  #TODO: f(x) ausgeben
    
    def derivative(self, x):
        '''TODO: Jakobi Matrix von f mit eingesetzten x wert... mit errors implementieren
        J = sym.zeros(len(self.f),len(self.vars))
        for i, fi in enumerate(self.f):
            for j, s in enumerate(self.vars):
                J[i,j] = sym.diff(fi, s)
        
        for i in range(J.lenth()):
            for j in range(J[i].lenth[]):
                J[i,j] = [J[i,j].replace(fr)]
        #TODO: auswerten mit x
        # z.b. so eingeben: Jacobian('u1 u2', ['2*u1 + 3*u2','2*u1 - 3*u2'])
        # vergleiche seite https://im-coder.com/berechnen-sie-die-jacobi-matrix-in-python.html'''
        #Ideen: siehe tests2
        pass


def newton_like(variant, f, x_0, dfz=None, x_1=None, tol=1e-8, max_iter=50):
    '''Abbruch: |xk-x(k-1)| < tol oder max_iter ist erreicht
    - assume x_0 is np.array'''
    if(f==None or x_0==None):
        raise ValueError('f or x_0 is given incorrectly!')
    
    x_k = x_0.copy()
    
    if variant=='standard':
        for i in range (max_iter):
            x_k1 = x_k - np.linalg.inv(f.derivative(x_k)) * f.__call__(x_k) #TODO: test errors...
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
                    return x_k1 #TODO: oder error werfen?
                x_k2 = x_k1 - ((x_k1 - x_k)/(f_sup))*f.__call__(x_k1)
                if np.linalg.norm(x_k1-x_k2) < tol:
                    return x_k2
                x_k = x_k1.copy()
                x_k1 = x_k2.copy()
            return x_k1
        # dim > 1
        for i in range (max_iter):
            x_k2 = x_k1 - (np.linalg.inv(f.derivative(x_k1))@f.__call__(x_k1))
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
