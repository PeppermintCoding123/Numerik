# --- Programmierblatt 05 Aufgabe 2 ---
import numpy as np
# %% --- rewrite this class to be correct with Vander, not whatever is going on now
class Vandermonde_model:
    #Interpolation with Newton
    def fit(self, x, y):
        if np.shape(x)[0] != np.shape(y)[0]:
            raise ValueError('x and y must have same dimension')
        n = np.shape(x)[0]
        if n<2: raise ValueError('at least two points must be given to plot graph')
        a = np.zeros(n)  
        
        a[0]=y[0]
        a[1]= (y[1]-y[0])/(x[1]-x[0])
        
        for j in range(2,n):
            Xj_k = x[j]-x[j-1]
            a[j] = -a[j-1]/Xj_k
            for k in range(2,j):
               Xj_k *= x[j]-x[j-k]
               a[j] -= a[j-k]/Xj_k
            a[j] += y[j]/Xj_k 
        
        self.x = x
        self.y = y
        self.p = a   
        
    def __call__(self, x):
        return np.polyval(self.p,x)
    
    def add_points(self, x, y):
        #Add aditional points from x & y without destroying existing p
        n = np.shape(self.p)
        if n<2: raise ValueError('at least two points must be given to plot graph')
        if np.shape(x)[0] != np.shape(y)[0]:
            raise ValueError('x and y must have same dimension')
        m = np.shape(x)[0]
        
        a = np.zeros(m) 
        x = np.concatenate(self.x, x)
        y = np.concatenate(self.y, y)
        
        for j in range(n,n+m): #TODO: Debugg this logic, might have errors
            Xj_k = x[j]-x[j-1]
            a[j-n] = -a[j-1]/Xj_k
            for k in range(2,j):
               Xj_k *= x[j]-x[j-k]
               if k > j:
                  a[j-n] -= self.p[n+j-k]/Xj_k #if ak value is in p
               else:
                   a[j-n] -= a[j-k]/Xj_k
            a[j-n] += y[j]/Xj_k 
        
        self.x = x
        self.y = y
        self.p = a 
# %%
class Lagrange_model:
    def fit(self, x, y):
        # fit polynom at x,y and save w_k
        if np.shape(x)[0] != np.shape(y)[0]:
            raise ValueError('x and y must have same dimension')
        n = np.shape(x)[0]
        if n<2: raise ValueError('at least two points must be given to plot graph')
        
        w = np.zeros(n) #--- is there a better way than a doubble-for-loop?
        for k in range(n):
            for i in range(n):
                w[k] /= (x[k]-x[i])
        self.w = w        
        self.x = x
        self.y = y
    
    def __call__(self, x):
        # x is 1D, so just 1 value
        n = self.x.shape[0]
        x_sum = 0
        y_sum = 0
        for k in range (n):
            x_sum += (self.w[k]/(x-self.x[k]))
            y_sum += (self.y[k]*self.w[k]/(x-self.x[k]))
        return y_sum/x_sum
    
    def add_points(self, x, y):
        #Add aditional points from x & y without destroying existing p
        n = np.shape(self.p)
        if n<2: raise ValueError('at least two points must be given to plot graph')
        if np.shape(x)[0] != np.shape(y)[0]:
            raise ValueError('x and y must have same dimension')
        m = np.shape(x)[0]
        
        a = np.zeros(m) 
        x = np.concatenate(self.x, x)
        
        for k in range(n,n+m):
            for i in range(n):
                a[k] /= (x[k]-x[i])
        
        self.w = np.concatenate(self.w, a)
        self.x = x
        self.y = np.concatenate(self.y, y)
        
# %%
class Newton_model:
    def fit(self, x, y):
        # fit polynom at x,y and save f0,...,f0_n
        if np.shape(x)[0] != np.shape(y)[0]:
            raise ValueError('x and y must have same dimension')
        n = np.shape(x)[0]
        if n<2: raise ValueError('at least two points must be given to plot graph')
        
        f = np.copy(y)
        ''' dependant on what itteration k in on, f[k] will either have 
            f0,...,i for i <= k (top row of diagram 5.18) or
            fi-k,...,i for i>k
        '''
        #TODO Debug & test if implementation is correct
        for k in range(1,n):
            for i in range(k,n):
                f[i]= (f[i]-f[i-1])/(x[i]-x[i-k])
        
        self.x = x
        self.y = y
        self.f = f
        
    def __call__(self, x):
        pass
    def add_points(self, x, y):
        pass

# %%
# Zeitvergleiche           
    
    
    
    
    
    
    
    
    
