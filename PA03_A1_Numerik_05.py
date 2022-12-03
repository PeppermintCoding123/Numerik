'''Numerik P3 Aufgabe 1'''
# %%
import numpy as np

# %%
def qr(A, mode='full', alg='Householder'):
    '''
    ---Implementation of Householder-Alghorhythem from Lecture---
    '''
    m, n = np.shape(A)

    if m < n:
        raise ValueError('Shape of A is wrong!')
    
    if alg == 'Householder':
        Q = np.eye(m, dtype=float)
        R = np.array(A.copy(), dtype=float)

        for j in range(0,n):
            v = R[j:m, j].copy()
            v[0] = v[0]-np.linalg.norm(v)
            v = np.array([v/np.linalg.norm(v)])
            Qj = np.eye(m-j) - ((np.sqrt(2)*(v.T)) @ (np.sqrt(2)*v))
            R[j:m,j:n] = Qj@R[j:m,j:n]  
            
            if mode!='R': Q[j:m, :] = Qj@Q[j:m,:]
            
        if mode=='full':
            return R, Q.T           
                       
        elif mode=='reduced':
            # return R (R^mxn)  and Q in (R^mxn)
            R = np.delete(R, range(n, m), 0)
            Q = np.delete(Q.T, range(n,m), 1) 
            
            return R, Q
        elif mode=='R':
            # return R (R^nxn)  
            R = np.delete(R, range(n, m), 0)
            return R
        else:
            raise ValueError('mode is unknown')
    else:
        pass

#%%
def backSubstitution(R, b):
    """backSubstitution
    ========
    Implementiert das Rueckwaertseinsetzten zum Loesen eines LGS.
    ----------
    Arguments:
        R (np.array): Untere Dreiecksmatrix.
        b (np.array): Rechte Seite.
    
    Returns:
        x (np.array): Loesungsvektor.
    """ 
    n = np.shape(R)[0]
    #check_input(R,b, criterions=['2-dim', 'quadratic', 'zero-diag', 'triangular'])
    x = np.zeros(n-1)
    x[-1] = b[-1] /  R[-1, -1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - R[i, i:]@x[i:]) / R[i, i]
    return x 

# %%
def ii(m, n):
    '''
    - x1 bis xm in [-3,3] erstelen (aequidistant vielleicht)
    - yi mit normalverteilung ausrechen & der Fromel
    => b = (y1,...ym)
    '''
    b = np.zeros(m)
    x = np.zeros(m)
    e = np.random.normal(0, 0.05, m)
    for i in range(m):
        x[i] = -3+ i*6/(m-1)
        b[i] = np.sin(3*x[i]) + x[i] + e[i]
    '''
    -mit QR von V & so weiter
    =>p
    '''
    V = np.vander(x, n)
    R, Q = qr(V)
    p = backSubstitution(R, Q.T@b)
    return p
    
#%%
'''
TODO: Visualisierung implementieren so wie polyfit aus Tutorium
'''
