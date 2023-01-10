import numpy as np
import matplotlib.pyplot as plt
import time

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
# %%

testrange = range(10, 201)
elapsedtime = np.zeros(np.size(testrange))
x = 7
for i, n in enumerate(testrange):
    
    p = np.random.rand(n)

    starttime = time.time()
    simple_polyval(x,p)
    elapsedtime[i] = time.time() - starttime
    
    #print('Iteration: ' + str(i))
  
print('Finished Calculation')
# %%
plt.close('all')
fig = plt.figure()
ax = plt.axes()

fig.suptitle('simple_polyval', fontsize=18)
plt.plot(testrange, elapsedtime)

plt.xlabel(r"Dimension des Graphen")
plt.ylabel(r"Berechnungszeit in Sekunden")

plt.show()

# %%

testrange = range(10, 201)
elapsedtime = np.zeros(np.size(testrange))
x = 7
for i, n in enumerate(testrange):
    
    p = np.random.rand(n)

    starttime = time.time()
    Horner_polyval(x,p)
    elapsedtime[i] = time.time() - starttime
    
    #print('Iteration: ' + str(i))
  
print('Finished Calculation')
# %%
plt.close('all')
fig = plt.figure()
ax = plt.axes()

fig.suptitle('Horner_polyval', fontsize=18)
plt.plot(testrange, elapsedtime)

plt.xlabel(r"Dimension des Graphen")
plt.ylabel(r"Berechnungszeit in Sekunden")
plt.show()

# %%

testrange = range(10, 201)
elapsedtime = np.zeros(np.size(testrange))
x = 27
for i, n in enumerate(testrange):
    
    p = np.random.rand(n)

    starttime = time.time()
    np.polyval(p,x)
    elapsedtime[i] = time.time() - starttime
    
    #print('Iteration: ' + str(i))
  
print('Finished Calculation')
# %%
plt.close('all')
fig = plt.figure()
ax = plt.axes()

fig.suptitle('numpy_polyval', fontsize=18)
plt.plot(testrange, elapsedtime)

plt.xlabel(r"Dimension des Graphen")
plt.ylabel(r"Berechnungszeit in Sekunden")

plt.show()