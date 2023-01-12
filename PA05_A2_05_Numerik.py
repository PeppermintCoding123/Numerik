# --- Programmierblatt 05 Aufgabe 2 ---
import numpy as np
import matplotlib.pyplot as plt
import time

# %%
'''TODO for Visitor: choose one of: 'fit', 'add_points' oder '__call__'
    to select which test is run. This metord allows a better view of graph.
'''
test = '__call__' 
# %%
def imput_check(t, dim_x, dim_y):
    if t == 'fit':
        if dim_x != dim_y:
            raise ValueError('x and y must have same dimension')
        if dim_x<2: raise ValueError('at least two points must be given to plot graph')
    if t == 'add':
        if dim_x != dim_y:
            raise ValueError('x and y must have same dimension')
# %%
class Vandermonde_model:
    def fit(self, x, y):
        n = np.shape(x)[0]
        imput_check('fit',n, np.shape(y)[0])
        v= np.vander(x,n, increasing=True)
        self.p = np.linalg.solve(v,y)
        self.x = x
        self.y = y
        
    def __call__(self,x):
        return np.polyval(self.p,x)
    
    def add_points(self, x, y):
        m = np.shape(x)[0]
        imput_check('add',m, np.shape(y)[0])
        
        self.y = np.hstack([self.y,y])
        self.x = np.hstack([self.x,x])
        
        self.p = np.linalg.solve(np.vander(self.x,np.shape(self.x)[0], increasing=True),self.y)
        

# %% ii)
class Lagrange_model:
    def fit(self, x, y):
        # fit polynom at x,y and save w_k
        n = np.shape(x)[0]
        imput_check('fit',n, np.shape(y)[0])
        
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
        n = np.shape(self.x)[0]
        m = np.shape(x)[0]
        imput_check('add',m, np.shape(y)[0])
        
        a = np.ones(m) 
        x = np.hstack([self.x,x])
        
        for k in range(n+m):
            if k<n:
                for i in range(m):
                    x[k] /= (x[k]-x[n+i])
            else:
                for i in range(n+m):
                    if i!=k:
                        a[k-n] /= (x[k]-x[i])
      
        self.w = np.hstack([self.w, a])
        self.x = x
        self.y = np.hstack([self.y,y])
        
# %% iii)
class Newton_model:
    def fit(self, x, y):
        # fit polynom at x,y and save f0,...,f0_n
        n = np.shape(x)[0]   
        imput_check('fit',n, np.shape(y)[0])
        f = np.copy(y)
        ''' dependant on what itteration k in on, f[k] will either have 
            f0,...,i for i <= k (top row of diagram 5.18) or
            fi-k,...,i for i>k
        '''
        #TODO Debug & test if implementation is correct
        for k in range(1,n):
            for i in range (k,n):
                f[k] = (f[i]-f[i-1])/(x[i]-x[i-1])
        
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
        n = np.shape(self.x)[0]
        m = np.shape(x)[0]
        imput_check('add',m, np.shape(y)[0])
        
        f = np.hstack([self.f,y])
        x = np.hstack([self.x,x])
        
        for k in range(1,n+m):
            if k < n:
                for i in range(n,m):
                    f[i]= (f[i]-f[i-1])/(x[i]-x[i-k])
            else:
                for i in range(k,m):
                    f[i]= (f[i]-f[i-1])/(x[i]-x[i-k])
        
        self.y = np.hstack([self.y,y])
        self.f = f
        
# %% iv)   Laufzeiten
def runntime_test(test):
    plt.close('all')
    fig, ax = plt.subplots()

    variants = ['Vandermonde_model', 'Lagrange_model', 'Newton_model']

    n = 50
    vm = Vandermonde_model
    lm = Lagrange_model
    nm = Newton_model
    res_vm = []
    res_lm = []
    res_nm = []


    ax.clear()
    ax.set_title('Runntime comparison ' + test)
    
    if test == 'fit':
        for i in range(2,n):
            x_dot = np.random.rand(i)
            y_dot = np.random.rand(i)
            for k, var in enumerate(variants):
                t_start = time.perf_counter()
                if var == 'Vandermonde_model':
                    vm.fit(vm, x_dot, y_dot)
                    res_vm.append(time.perf_counter() - t_start)
                elif var == 'Lagrange_model':
                    lm.fit(lm, x_dot, y_dot)
                    res_lm.append(time.perf_counter() - t_start)
                elif var == 'Newton_model':
                    nm.fit(nm, x_dot, y_dot)
                    res_nm.append(time.perf_counter() - t_start)
        x_seite = np.arange(2,n)
                    
    elif test == 'add_points':
        x_dot = np.random.rand(20)
        y_dot = np.random.rand(20)
        vm.fit(vm, x_dot, y_dot)
        lm.fit(lm, x_dot, y_dot)
        nm.fit(nm, x_dot, y_dot)
        
        for i in range(1,n):
            x_dot = np.random.rand(i)
            y_dot = np.random.rand(i)
            for k, var in enumerate(variants):
                t_start = time.perf_counter()
                if var == 'Vandermonde_model':
                    vm.add_points(vm, x_dot, y_dot)
                    res_vm.append(time.perf_counter() - t_start)
                elif var == 'Lagrange_model':
                    lm.add_points(lm, x_dot, y_dot)
                    res_lm.append(time.perf_counter() - t_start)
                elif var == 'Newton_model':
                    nm.add_points(nm, x_dot, y_dot)
                    res_nm.append(time.perf_counter() - t_start)
        x_seite = np.arange(21,n+20)
                    
    elif test == '__call__':
        for i in range(2,n):
            x_dot = np.random.rand(i)
            y_dot = np.random.rand(i)
            vm.fit(vm, x_dot, y_dot)
            lm.fit(lm, x_dot, y_dot)
            nm.fit(nm, x_dot, y_dot)
            x = np.random.uniform(-100, 100)
            
            for k, var in enumerate(variants):
                t_start = time.perf_counter()
                if var == 'Vandermonde_model':
                    vm.__call__(vm,x)
                    res_vm.append(time.perf_counter() - t_start)
                elif var == 'Lagrange_model':
                    lm.__call__(lm,x)
                    res_lm.append(time.perf_counter() - t_start)
                elif var == 'Newton_model':
                    nm.__call__(nm,x)
                    res_nm.append(time.perf_counter() - t_start)
        x_seite = np.arange(2,n)
    

    plt.plot(x_seite, res_vm, color='blue', label='Vandermonde_model')
    ax.plot(x_seite, res_lm, color='yellow', label='Lagrange_model')
    ax.plot(x_seite, res_nm, color='red', label='Newton_modely') 
    ax.legend()   
    
    plt.xlabel("dimension of n")
    plt.ylabel("runntime in sec")

    plt.show()
# %% running test
runntime_test(test)