import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider
#Slider benutzt du nicht, insofern würde ich ihn vielleicht nicht importieren
import time

print('Try 1')

# %% i)
def simple_polyval(x,p):
    '''arrumed: x is value and p is Vektors (np.arrays) '''
    n = np.shape(p)[0]
    if n==0: return 0 #deg = 0
    res = p[n-1]
    x_drag = x
    for i in range(2,n+1):
        res = (p[n-i] * x_drag)
        x_drag *= x
    return res
    
# %% ii)
def Horner_polyval(x,p):
    '''arrumed: x is value and p is Vektors (np.arrays) '''
    n = np.shape(p)[0]
    if n==0: return 0 #deg = 0
    res = p[0]
    for i in range(1,n):
        res *= x
        res += p[i]
    return res

# %% iii)
plt.close('all')
fig, ax = plt.subplots()

variants = ['simple_polyval', 'Horner_polyval', 'numpy']
colors = {'simple_polyval':'blue', 'Horner_polyval':'yellow', 'numpy':'red'}

n = 20
x = 2
#p = np.random.rand(n,1)
res_sp = []
res_Hp = []
res_p = []
'''
res_sp = np.zeros(n)
res_Hp = np.zeros(n)
res_p = np.zeros(n)
'''


ax.clear()
ax.set_title('Laufzeit Vergleich')

for i in range(n):
    p = np.random.rand(i,1)
    for k, var in enumerate(variants):
        t_start = time.perf_counter()
        if var == 'simple_polyval':
            simple_polyval(x,p)
            #res_sp = time.perf_counter() - t_start
            res_sp.append(time.perf_counter() - t_start)
        elif var == 'Horner_polyval':
            Horner_polyval(x,p)
            #res_Hp = time.perf_counter() - t_start
            res_Hp.append(time.perf_counter() - t_start)
        elif var == 'numpy':
            np.polyval(p, x)
            #res_p = time.perf_counter() - t_start
            res_p.append(time.perf_counter() - t_start)
 
x_seite = np.arange(1,n+1,1)

plt.plot(x_seite, res_sp, color='blue', label='simple_polyval')
ax.plot(x_seite, res_Hp, color='yellow', label='Horner_polyval')
ax.plot(x_seite, res_p, color='red', label='numpy') 
ax.legend()   
   
plt.show()