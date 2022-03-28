import numpy as np
from tqdm import trange
from numba import njit

n = 500
dt = 0.01 
time = np.linspace(0,dt*n,n)
T = 180/119.7


def placement(n_,d):

    N = 4*n_**3
    L = d*n_
    a = np.zeros((N,n,3))
    v = np.zeros(a.shape)
    r = np.zeros((N,n,3))
    
    index = 0
    for i in range(n_):
        for j in range(n_):
            for k in range(n_):
                r[index,0,:] = d*np.array([i,j,k])
                index += 1
                r[index,0,:] = d*np.array([i,j+0.5,k+0.5])
                index += 1
                r[index,0,:] = d*np.array([0.5+i,j,0.5+k])
                index += 1
                r[index,0,:] = d*np.array([0.5+i,0.5+j,k])
                index += 1

    v[:, 0] = np.random.normal(0, np.sqrt(T), size=(N, 3))
    return a,v,r,N
a,v,r,N= placement(4,1.7)



@njit
def calc_acc_pbc(r, cutoff):
    """Initialize acceleration
    """
    cutoff_sqrd = cutoff * cutoff
    a = np.zeros(r.shape)
    u = 0
    u_c = 4*(cutoff_sqrd**(-6)-cutoff_sqrd**(-3))
    for i in range(r.shape[0]):
        for j in range(i+1, r.shape[0]):       
            dr = r[i] - r[j]
            dist2 = np.sum(dr**2)                             # find the shortest distance between
            if dist2 < cutoff_sqrd:                           # particles, might be through the wall
                dist6inv = dist2**(-3)
                dist12inv = dist6inv**2
                aij = (2*dist12inv-dist6inv) * dr / dist2
                a[i] += aij
                a[j] -= aij
                u += 4*(dist12inv-dist6inv)-u_c
    return 24 * a, u


U = np.zeros(n)
def func(i, a, v, r):

    r[:,i+1,:] = r[:,i,:] + v[:,i,:]*dt + 0.5*a[:,i,:]*dt**2

    a_new, u = calc_acc_pbc(r[:, i+1], 3)
    U[i+1] = u
   
    a[:, i+1] = a_new

    v[:,i+1,:] = v[:,i,:] + 0.5*(a[:,i,:] + a[:,i+1,:])*dt

   


a_new, u = calc_acc_pbc(r[:, 0], 3)
a[:, 0] = a_new

for i in trange(n-1):
    func(i, a, v, r)


def K(v):
    return 0.5*np.sum(v**2,axis=(0,2))


t = np.linspace(0,dt*n,n)

import matplotlib.pyplot as plt
plt.plot(t,U,label='Potential')
plt.plot(t,K(v),label='Kinetic')
plt.plot(t,K(v)+U,label='Total')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.show()



file = open('Project.txt','w')
for i in range(n):

    file.write(f'{N} \n')
    file.write("Type            x                y              z \n")
    for j in range(N):  
        file.write(f'Ar    {r[j,i,0]}     {r[j,i,1]}      {r[j,i,2]}   \n')
       
file.close()


