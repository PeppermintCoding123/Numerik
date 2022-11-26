'''Numerik P3 Aufgabe 1'''
# %%
import numpy as np

# %%
def qr(A, mode='full', alg='Householder'):
    '''
    ---Implementation of Householder-Alghorhythem from Lecture---
    TODO:
        - test m >= n for A is R^mxn -Matrix
        - never Calculate Householder-Matrix directly
        - adjust calculations according to Keyword
    '''
    m, n = np.shape(A)

    if m < n:
        raise ValueError('Shape of A is wrong!')
    
    if mode=='full':
        # return R (R^mxn)  and Q in (R^mxm)
        w = np.zeros(m)
        R = A
        
        for j in range(0,n-1):
            v = R[j:m-1, j]
            v[0] = v[0] - np.sqrt(np.sum(v))
            v = np.sqrt(np.sum(v)) * v
            
            I = eye(m-j-1)
            R[j:-1,j:n-1] = (I-2*(v@v.T))@R[j:m-1,j:n-1]
            w = np.hstack((w[0:j], w[j:m-1] * v)
            
        return R, (np.eye(m)- 2*(w@w.T) #--How to correct this syntax?
                   
    else if mode=='reduced':
        # return R (R^mxn)  and Q in (R^mxn) --I do not know how to get Q.
    else if mode=='R':
        # return R (R^nxn)
        R = A
        
        for j in range(0,n-1):
            v = R[j:m-1, j]
            v[0] = v[0] - np.sqrt(np.sum(v))
            v = np.sqrt(np.sum(v)) * v
            
            I = eye(m-j-1)
            R[j:-1,j:n-1] = (I-2*(v@v.T))@R[j:m-1,j:n-1]
            
        return R
    else:
        raise VlaueError('mode is unknown')

# %%