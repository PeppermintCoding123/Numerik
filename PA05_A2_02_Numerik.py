# --- Programmierblatt 05 Aufgabe 2 ---
import numpy as np
# %%
class Vander_model:
    def fit(self, x, y):
        if np.shape(x)[0] != np.shape(y)[0]:
            raise ValueError('x and y must have same dimension')
        n = np.shape(x)[0]
        if n<2: raise ValueError('at least two points must be given to plot graph')
        self.p = np.linalg.solve(np.vander(x,n, increasing=True),y)
        
    def __call__(self,x):
        return np.polyval(self.p,x)
    
    def add_points(self, x, y):
        #Add aditional points from x & y without destroying existing p
        n = np.shape(self.p)
        if n<2: raise ValueError('at least two points must be given to plot graph')
        if np.shape(x)[0] != np.shape(y)[0]:
            raise ValueError('x and y must have same dimension')
        m = np.shape(x)[0]

        #--- add points but kepe the existing Polynom p?
# %% ii)
class Lagrange_model:
    def fit(self, x, y):
        # fit polynom at x,y and save w_k
        if np.shape(x)[0] != np.shape(y)[0]:
            raise ValueError('x and y must have same dimension')
        n = np.shape(x)[0]
        if n<2: raise ValueError('at least two points must be given to plot graph')
        
        w = np.ones(n) #--- is there a better way than a doubble-for-loop?
        for k in range(n):
            for i in range(n):
                if i!=k:
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
        
        a = np.ones(m) 
        x = np.concatenate(self.x, x)
        
        for k in range(n,n+m):
            for i in range(n):
                a[k] /= (x[k]-x[i])
        
        self.w = np.concatenate(self.w, a)
        self.x = x
        self.y = np.concatenate(self.y, y)
        
# %% iii)
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
        # x is 1D, so just 1 value
        N_k = 1
        p_x = 0
        for k in range(np.shape(self.x)[0]):
            p_x += self.f[0]*N_k
            N_k *= (x-self.x[k])
        return p_x
    
    def add_points(self, x, y):
        #Add aditional points from x & y without destroying existing p
        n = np.shape(self.p)
        if n<2: raise ValueError('at least two points must be given to plot graph')
        if np.shape(x)[0] != np.shape(y)[0]:
            raise ValueError('x and y must have same dimension')
        m = np.shape(x)[0]
        
        f = np.concatenate(self.f, y)
        x = np.concatenate(self.x, x)
        
        for k in range(1,n+m):
            if k < n:
                for i in range(n,m):
                    f[i]= (f[i]-f[i-1])/(x[i]-x[i-k])
            else:
                for i in range(k,m):
                    f[i]= (f[i]-f[i-1])/(x[i]-x[i-k])
        
        self.y = np.concatenate(self.y, y)
        self.f = f
        

# %%
# Debugg  
x = np.array([0,1,3])
y = np.array([1,3,2])

vm = Vander_model
#print(vm.fit(vm, x,y))
lm = Lagrange_model
nm = Newton_model
print(nm.fit(nm,x,y))
    
# %% iv)   
    
    
    
    
    
    