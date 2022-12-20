

import numpy as np

# %% 
class f1:
    def __call__(self, y):
        return 1/np.tan(y[0])
    def derivative(self, y):
        return np.array([[-1/((np.sin(float(y[0])**2)))]])

class f2:
    def __call__(self, y):
        return np.array([np.sin(y[0]),np.cos(y[1])])
    def derivative(self, y):
        return np.array([[np.cos(y[0][0]),0],[0,-np.sin(y[1][0])]])

class f3:
    def __call__(self, y):
        tmp = np.zeros(len(y))
        for i in range(len(y)):
            tmp[i] = np.exp(-(y[i][0]**2)) -1
        return tmp
    def derivative(self, y):
        tmp = np.zeros((len(y),len(y)))
        for i in range(len(y)):
            tmp[i][i] = np.exp(-((y[i][0])**2))*(-2*(y[i][0]))
        return tmp

# %%
def newton_like(f, x_0, dfz=None, x_1=None, tol=1e-8, max_iter=50, delta=0.01, variant='standard'):
    '''Abbruch: |xk-x(k-1)| < tol oder max_iter ist erreicht'''
    if(f is None or x_0 is None):
        raise ValueError('f or x_0 is given incorrectly!')
    
    x_k = x_0.copy()
    n = np.shape(x_0)[0]
    
    if variant=='standard':
        f = f()
        for i in range (max_iter):
            x_k1 = x_k - np.linalg.inv(f.derivative(x_k)) * f.__call__(x_k) #für die angegebenen Testfunktionen, Matrix immer quadratisch --> np.linalg.inv genügt, nicht np.linalg.pinv nötig
            if i == max_iter - 1:
                print('---Das ist x_k' + str(i) + '---')
                return x_k1
            if np.linalg.norm(x_k-x_k1) < tol:
                print('---Das ist x_k' + str(i) + '---')
                return x_k1
            x_k = x_k1.copy()
        return x_k
    
    elif variant=='secant':
        if f == f1:
            if x_1 is None:
                raise ValueError('In Variant==secant: x_1 must be given!')
            x_k1 = x_1.copy()
            for i in range (max_iter):
                f_sup = f.__call__(x_k1) - f.__call__(x_k)
                if f_sup == 0:
                    return x_k1 
                x_k2 = x_k1 - ((x_k1 - x_k)/(f_sup))*f.__call__(x_k1)
                if np.linalg.norm(x_k1-x_k2) < tol:
                    return x_k2
                x_k = x_k1.copy()
                x_k1 = x_k2.copy()
            return x_k1
        else:
            for k in range(max_iter):
                e = np.zeros((n,1))
                e[0] = delta
                Jk = (1/delta) * (f.__call__(x_k + e) - f.__call__(x_k))
                e[0] = 0
                for j in range(1,n):
                    e[j] = delta
                    Jkj = (1/delta) * (f.__call__(x_k + e) - f.__call__(x_k))
                    Jk = np.hstack((Jk, Jkj))
                    e[j] = 0
                x_k1 = x_k - np.linalg.solve(Jk, (Jk@x_k - f.__call__(x_k))) #Uns wurde gesagt, wir sollen nicht np.linalg.inv(Jk) benutzen, sondern das Ganze mittels eines anderen Verfahrens berechnen
                x_k = x_k1
            return x_k
    
    if variant=='simple':
        if dfz is None:
            raise ValueError('In variant== simple: dfz is not given!')
        dfzinv = np.linalg.inv(dfz)
        for i in range (max_iter):
            x_k1 = x_k - (dfzinv * f.__call__(x_k))
            if i == max_iter - 1:
                print('---Das ist x_k' + str(i) + '---')
                return x_k1
            if np.linalg.norm(x_k-x_k1) < tol:
                print('---Das ist x_k' + str(i) + '---')
                return x_k1
            x_k = x_k1.copy()
        return x_k
    
    else:
        raise ValueError('variant is not known!')
    
   
# %% Teilaufgabe (iv)

print(newton_like(f1, np.array([1]), variant='standard'))
print(newton_like(f2, np.array([[0],[1]]), variant='standard'))
print(newton_like(f3, np.ones((37,1)), variant='standard'))


print(newton_like(f1, np.array([1]), x_1 = np.array([1]) + 0.01, variant='secant'))
print(newton_like(f2, np.array([[0],[1]]), variant='secant'))
print(newton_like(f3, np.ones((37,1)), variant='secant'))


#dfz für f1
y1 = np.array([1])
f1 = f1()
dfz1 = f1.derivative(y1)

print(newton_like(f1, np.array([1]), dfz1, variant='simple'))

#dfz für f2
y2 = np.array([[0],[1]])
f2 = f2()
dfz2 = f2.derivative(y2)

print(newton_like(f2, np.array([[0],[1]]), dfz2, variant='simple'))

#defz für f3
y3 = np.ones((37,1))
f3 = f3()
dfz3 = f3.derivative(y3)

print(newton_like(f3, np.ones((37,1)), dfz3, variant='simple'))
