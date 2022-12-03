#%% Ausfuerliche Version von QR in Python bezogen auf Theorie06 Aufgabe 1

import numpy as np


A = np.array([[0,1,1],[0,0,1],[1,0,1],[0,0,1]])
m, n = np.shape(A)
Q = np.eye(m, dtype=float)
R = np.array(A.copy(), dtype=float)


for j in range(0,n):
    v = R[j:m, j].copy()
    c = np.linalg.norm(v)
    d = float(v[0])
    v[0] = d-c
    print(v)
    v = v/np.linalg.norm(v)
    v = np.array([v])
    B = (np.sqrt(2)*(v.T)) @ (np.sqrt(2)*v)
    Qj = np.eye(m-j) - B
    Q[j:m, :] = Qj@Q[j:m,:]
    R[j:m,j:n] = Qj@R[j:m,j:n]


print(Q.T)
print(R)

