
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time

#%% Aufgabe 1
#(i)
def simple_polyval(x,p):
    'Annahme: x Wert, p Vektor'
    m,n = np.shape(p)
    if m == 0:
        return 0
    P = 0
    for i in range(n):
        P += p[i]*(x**i)
    return P

#(ii)
def Horner_polyval(x, a):
    m,n = np.shape(a)
    if m == 0:
        return 0
    P = a[n-1]*x + a[n-2]
    for i in range(2,n):
        P *= x+ a[n-1-i]
    return P

#(iii) Laufzeiten vergleichen
plt.close('all')
fig, ax = plt.subplots(1,1)

variants = ['simple_polyval', 'Horner_polyval', 'numpy']
colors = {'simple_polyval':'blue', 'Horner_polyval':'cyan', 'numpy':'black'}

axn = fig.add_axes([0.25, 0.0, 0.65, 0.03])
n_slider = Slider(
    ax = axn, 
    label='Polynom-Grad',
    valmin = 0,
    valmax = 100,
    valinit = 3,
    valstep = 1.
    )
axx = fig.add_axes([0.25, -0.1, 0.65, 0.03])
x_slider = Slider(
    ax = axx,
    label='Wert für x',
    valmin = 0,
    valmax = 20,
    valinit=5,
    valstep=0.5
    )

def update(val):
    n = int(n_slider.val)
    x = int(x_slider.val)
    
    p = np.random.rand(n,1)
    
    ax.clear()
    ax.set_title('Laufzeit für Polynom-Grad = ' + str(n))
    
    excec_times = {}
    for i, var in enumerate(variants):
        t_start = time.perf_counter()
        if var == 'simple_polyval':
            simple_polyval(x,p)
        elif var == 'Horner_polyval':
            Horner_polyval(x,p)
        elif var == 'numpy':
            np.polyval(p, x)
            
        excec_times[var] = time.perf_counter() - t_start
        ax.bar(i, excec_times[var], color = colors[var], label = variants[i])
    
    ax.legend()
    
update(0)
n_slider.on_changed(update)
x_slider.on_changed(update)
    
    

    
    