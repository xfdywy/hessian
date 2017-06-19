import numpy as np
import scipy
import scipy.linalg as la
import matplotlib.pyplot as plt

###### some number
dim_n = 20
test_n = 60

def diagmatrix(dim,positive):
    temp = np.random.uniform(0,10,[dim])
    temp[positive:]  *= -1
    return(np.diag(temp))






A = np.random.random([dim_n,dim_n])
V = la.orth(A)

B = diagmatrix(dim_n,17)

mat = np.dot(np.dot(np.transpose(V),B),V)

res = np.zeros(test_n)
for ii in range(test_n):
    testvec = np.random.randn(dim_n,1)

#    testvec = np.zeros([dim_n,1])
#    testvec[ii  % dim_n] =1
    
    res[ii] = np.dot(np.transpose(testvec) , mat).dot(testvec)
    

plt.plot(res)
print(min(res))