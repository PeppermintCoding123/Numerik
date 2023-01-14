#PB 06 Aufgabe 1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
# %%
''' Questions:
    - is f a special np.function or should we code in f ourselves?
'''
# %%
class f:
    def __call__(x):
        return np.sin(x) * (x+1)
    
    def integ(a,b):#used to compare accuracy of approximations
        return -np.sin(a)+(a+1)*np.cos(a) + np.sin(b)-(b+1)*np.cos(b)

def __apply(f,a,b,n, w):
    s = 0
    for i in range(n):
       s += f.__call__(a + i/(n-1)*(b-a)) *w[i]
    return s
    
# %%
def quad(f, a=0., b=1., method='trapezoidal', num_ints=1):
    if num_ints == 1:
        if method=='trapezoidal':#Bsp 6.5.1
            return __apply(f,a,b,2,np.array([1,1])) / 2
            #return (f.__call__(a) + f.__call(b))*(b-a)/2
        if method=='Simpson':#Bsp 6.5.2
            return __apply(f,a,b,3,np.array([1,4,1])) / 3 
            #return (f.__call__(a) + 4*f.__call__((a+b)/2) * f.__call__(b))
        if method=='pulcherrima':
            return __apply(f,a,b,4,np.array([1,3,3,1])) /8
        if method=='Milne':
            return __apply(f,a,b,5,np.array([7,32,12,32,1]))/90
        if method=='six-point':
            return __apply(f,a,b,6,np.array([19,75,50,50,75,19]))/228
        if method=='Weddle':
            return __apply(f,a,b,7,np.array([41,216,27,272,27,216,41]))/140
    if num_ints >1:
        s =0
        for j in range(num_ints):
            s += quad(f, a, (a + j*(b-a)/num_ints), method, 1)
        return s
    else:
        raise ValueError('num_ints must be >= 1')


#print(quad(f,2,7, 'Weddle',7))
# %% Accuaracy
plt.close('all')
fig, ax = plt.subplots()
plt.tight_layout(pad=4)

n = 10
variants = ['trapezoidal', 'Simpson', 'pulcherrima', 'Milne', 'six-point', 'Weddle']
colors = {'trapezoidal':'springgreen', 'Simpson':'turquoise', 'pulcherrima':'deepskyblue', 'Milne':'mediumslateblue', 'six-point':'mediumorchid','Weddle':'violet'}

axa = fig.add_axes([0.2, 0.1, 0.65, 0.03])
a_slider = Slider(
    ax = axa, 
    label='a:',
    valmin = -10,
    valmax = 10,
    valinit = 2,
    valstep = 0.1
    )

axb = fig.add_axes([0.2, 0.05, 0.65, 0.03])
b_slider = Slider(
    ax = axb,
    label='b:',
    valmin = -10,
    valmax = 10,
    valinit = 5,
    valstep = 0.1
    )

def update(val):
    ax.clear()
    a = int(a_slider.val)
    b = int(b_slider.val)  
    
    ax.clear()
    ax.set_title('Integral approximation')
    
    n_seite = np.arange(1,n+1)
    for var in variants:
        integ = np.zeros(n-1)
        for i in range(n-1):
            integ[i] = quad(f,a,b,var,i+1)
        ax.plot(n_seite[:-1], integ, color=colors[var], label=var)
    
    integ = np.empty(n-1) 
    integ.fill(f.integ(a, b))
    ax.plot(n_seite[:-1], integ, color='black', label='actual')
    
    ax.legend()
    #ax.xlabel('num_ints') #---How to fix lables?
    #ax.ylabel('integral result')  
        #ax.legend()

update(0)  
a_slider.on_changed(update)
b_slider.on_changed(update)
  







    