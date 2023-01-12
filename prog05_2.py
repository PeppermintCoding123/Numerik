"""
Aufgabe 2
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time

#aus Aufgabe 1
def Horner_polyval(x, a):
    m,n = np.shape(a)
    if m == 0:
        return 0
    P = a[n-1]*x + a[n-2]
    for i in range(2,n):
        P *= x 
        P += a[n-1-i]
    return P

#%%Aufgabe 2

class Vandermonde_model:
    def fit(self, x, y):
        self.x = x
        self.y = y
        if np.shape(self.x)[0] != np.shape(self.y)[0]:
            raise ValueError('shape of x and y should be the same')
        n = np.shape(self.x)[0]
        self.p = np.linalg.solve(np.vander(self.x,n,increasing=True), self.y)
    
    def __call__(self, x):
        return Horner_polyval(x, self.p)
    
    def add_points(self, x, y):
        self.x = x
        self.y = y
        if np.shape(self.x)[0] != np.shape(self.y)[0]:
            raise ValueError('shape of x and y should be the same')
        n = np.shape(self.x)[0]
        self.p = np.linalg.solve(np.vander(self.x,n,increasing=True), self.y)

class Lagrange_model:
    def fit(self, x, y):
        self.x = x
        self.y = y
        if np.shape(self.x)[0] != np.shape(self.y)[0]:
            raise ValueError('shape of x and y should be the same')
        for k in range(np.shape(self.x)[0]):
            self.__w_k(self.x, k)
    
    def __w_k(self, x, k):
        w_k = 1
        for i in range(np.shape(x)[0]):
            if i==k:
                pass
            else:
                w_k *= (x[k] - x[i])**(-1)
        self.w_k = w_k
        return self.w_k
    
    def __call__(self, x):
        a=0
        for k in range(np.shape(self.x)[0]):
            a += (self.y[k]*(self.__w_k(self.x,k)))/(x - self.x[k])
        b=0
        for k in range(np.shape(self.x)[0]):
            b += self.__w_k(self.x,k)/(x - self.x[k])
        return a/b
    
    def add_points(self, x, y):
        x_new = np.hstack((self.x, x))
        y_new = np.vstack((self.y, y))
        for k in range(np.shape(x_new)[0]):
            if k in range(np.shape(self.x)[0]):
                for i in range(np.shape(x)[0]):
                    self.w_k *= (self.x[k] - x[i])**(-1)
            else:
                self.w_k = self.__w_k(x_new, k)
        self.x = x_new
        self.y = y_new
        

class Newton_model:
    def fit(self, x, y):
        self.x = x
        self.y = y
        if np.shape(self.x)[0] != np.shape(self.y)[0]:
            raise ValueError('shape of x and y should be the same')
        self.n = np.shape(self.x)[0]
        coeff = []
        f_ = np.zeros((self.n-1,1))
        f = self.y.copy()
        for i in range(self.n-1):
            for j in range(self.n-2-i):
                f_[j] = (f[j+1]-f[j])/(self.x[j+1+i] - self.x[j])
                if j == 0:
                    coeff.append(f_[j])
                f = f_.copy()
        self.coeff = coeff
        
    '''  nicht mit dividierten Differenzen
    def __N_k(self, x, k):
        if k == 0:
            return self.N_k = 1
        else:
            N_k = 1
            for i in range(k):
                N_k *= (x-self.x[i])
            return self.N_k
    
    def __a:
        n = self.x[0]
        L = np.zeros(n, n)
        for i, j in range(n):
            L[i,j] = self.__N_k(self.x[i], j)
        self.a = np.linalg.solve(L, self.y)
        return self.a
     ''' 
    
    def __call__(self, x):
        p = self.coeff[-2] + self.coeff[-1]*(x-self.x[-1])
        for i in range(2,self.n-1):
            p *= (x - self.x[-i])
            p += self.coeff[-i]
        self.p = p
    
    def add_points(self, x, y):
        if np.shape(x)[0] != np.shape(y)[0]:
            raise ValueError('shape of x and y should be the same')
        x_new = np.hstack((self.x, x))
        y_new = np.vstack((self.y, y))
        n_new = np.shape(x_new)[0]
        
        
        f_ = np.zeros((n_new-2,1))
        f = y_new.copy()
        for i in range(1,n_new-1):
            if i <= self.n:
                for j in range(1,n_new-2-i):
                    f_[j] = (f[j+1]-f[j])/(x_new[j+1+i] - x_new[j])
                    f = f_.copy()
            if i > self.n:
                for j in range(n_new-2-i):
                    f_[j] = (f[j+1]-f[j])/(x_new[j+1+i] - x_new[j])
                    if j == 0:
                        self.coeff.append(f_[j])
                    f = f_.copy()
        
        self.coeff = self.coeff
        self.x = x_new
        self.y = y_new


#Laufzeiten

plt.close('all')
fig, ax = plt.subplots(1,3,sharex=True, sharey=True)
plt.tight_layout(pad=4)

variants = ['Vandermonde_model', 'Lagrange_model', 'Newton_model']
colors = {'Vandermonde_model':'blue', 'Lagrange_model':'cyan', 'Newton_model':'black'}

axx = fig.add_axes([0.2, 0.05, 0.65, 0.03])
x1_slider = Slider(
    ax = axx,
    label='Wert für x',
    valmin = 0,
    valmax = 20,
    valinit=5,
    valstep=0.5
    )

axn = fig.add_axes([0.2, 0.1, 0.65, 0.03])
n_slider = Slider(
    ax = axn, 
    label='Anzahl an Stützstellen',
    valmin = 0,
    valmax = 100,
    valinit = 10,
    valstep = 1.
    )
 

def update(val):
    x1 = int(x1_slider.val)
    n = int(n_slider.val)
    
    
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    ax[0].set_title('Laufzeit Polynom-Fitting')
    ax[1].set_title('Auswertung an beliebigen Punkten')
    ax[2].set_title('Re-Fitting mit zusätzlichen Stützstellen')
    
    van = Vandermonde_model()
    lag = Lagrange_model()
    newton = Newton_model()
    
    
    x = np.random.normal(0, 0.1, (np.linspace(-2,2,n)).shape)
    y = np.random.rand(n,1)
    x_ex = np.random.normal(0, 0.1, (np.linspace(2,3,5)).shape)
    y_ex = np.random.rand(5,1)
    #Polynom-Fitting
    excec_times = {}
    for i, var in enumerate(variants):
        t_start = time.perf_counter()
        if var == 'Vandermonde_model':
            van.fit(x, y)
        if var == 'Lagrange_model':
            lag.fit(x, y)
        if var == 'Newton_model':
            newton.fit(x, y)
    
        excec_times[var] = time.perf_counter() - t_start
        ax[0].bar(i, excec_times[var], color = colors[var], label = variants[i])
        
    ax[0].legend()
    
    #Auswertung an beliebigen Punkten
    excec_times = {}
    for i, var in enumerate(variants):
        t_start = time.perf_counter()
        if var == 'Vandermonde_model':
            van.__call__(x1)
        if var == 'Lagrange_model':
            lag.__call__(x1)
        if var == 'Newton_model':
            newton.__call__(x1)
    
        excec_times[var] = time.perf_counter() - t_start
        ax[1].bar(i, excec_times[var], color = colors[var], label = variants[i])
    
    ax[1].legend()
    
    #Re-Fitting mit zusätzlichen Stützstellen
    excec_times = {}
    for i, var in enumerate(variants):
        t_start = time.perf_counter()
        if var == 'Vandermonde_model':
            van.add_points(x_ex, y_ex)
        if var == 'Lagrange_model':
            lag.add_points(x_ex, y_ex)
        if var == 'Newton_model':
            newton.add_points(x_ex, y_ex)
    
        excec_times[var] = time.perf_counter() - t_start
        ax[2].bar(i, excec_times[var], color = colors[var], label = variants[i])
        
    ax[2].legend()
    
    
update(0)
x1_slider.on_changed(update)  
n_slider.on_changed(update)
    
    
            
