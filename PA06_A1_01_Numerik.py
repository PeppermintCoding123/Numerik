#PB 06 Aufgabe 1
import numpy as np
# %%
''' Questions:
    - is f a special np.function or should we code in f ourselves?
'''
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
        for j in num_ints:
            s += quad(f, a, (a + j*(b-a)/num_ints), method, num_ints=1)
    else:
        raise ValueError('num_ints must be >= 1')
# %%
    