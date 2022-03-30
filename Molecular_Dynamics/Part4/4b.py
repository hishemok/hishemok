import numpy as np
from tqdm import trange
from numba import njit

n = 500
dt = 0.01 
time = np.linspace(0,dt*n,n)
T = 167.4/119.7


def placement(n_,d):
    N = 4*n_**3
    L = n_*d
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
    return a,v,r,N,L
a,v,r,N,L = placement(4,1.7)

@njit
def calc_acc_pbc(r, cutoff, L):
    """Initialize acceleration
    """
    cutoff_sqrd = cutoff * cutoff
    a = np.zeros(r.shape)
    u = 0
    u_c = 4*(cutoff_sqrd**(-6)-cutoff_sqrd**(-3))
    for i in range(r.shape[0]):
        for j in range(i+1, r.shape[0]):       
            dr = r[i] - r[j]
            dr -= np.round(dr/L, 0, np.empty_like(dr)) * L
            dist2 = np.sum(dr**2)                             # find the shortest distance between
            if dist2 < cutoff_sqrd:                           # particles, might be through the wall
                dist6inv = dist2**(-3)
                dist12inv = dist6inv**2
                aij = (2*dist12inv-dist6inv) * dr / dist2
                a[i] += aij
                a[j] -= aij
                u += 4*(dist12inv-dist6inv)-u_c
    return 24 * a, u



@njit
def func(i, a, v, r, L):

    r[:,i+1,:] = r[:,i,:] + v[:,i,:]*dt + 0.5*a[:,i,:]*dt**2
    r[:,i+1,:] = r[:,i+1,:] - np.floor(r[:,i+1,:] / L) * L

    a_new, u = calc_acc_pbc(r[:, i+1], 3, L)
   
    a[:, i+1] = a_new
    
    v[:,i+1,:] = v[:,i,:] + 0.5*(a[:,i,:] + a[:,i+1,:])*dt

  

def Auto(v):
    over = np.sum(np.transpose(np.einsum('ijk,ik->ij', v, v[:,0,:])),axis=1)
    under = np.sum(np.einsum('ij,ij->i', v[:,0,:], v[:,0,:]))
    return 1/N*over/under


a_new, u = calc_acc_pbc(r[:, 0], 3, L)
a[:, 0] = a_new

for i in trange(n-1):
    func(i, a, v, r, L)

import matplotlib.pyplot as plt
plt.plot(time,Auto(v),label='Autocorrelation')
plt.xlabel('Time')
plt.ylabel('A(t)')
plt.legend()
plt.show()

#this is what i used to average it out to create a smoother plot
#as well as finding the diffusion coeffisient

'''
Average_auto = np.zeros((25,n))

for run in range(25):

    print(run)
    a_new, u = calc_acc_pbc(r[:, 0], 3, L)
    a[:, 0] = a_new

    for i in trange(n-1):
        func(i, a, v, r, L)
    
    v[:,0,:] = v[:,-1,:]
    r[:,0,:] = r[:,-1,:]

    a_new, u = calc_acc_pbc(r[:, 0], 3, L)
    a[:, 0] = a_new

    for i in trange(n-1):
        func(i, a, v, r, L)

    Average_auto[run] = Auto(v)
    


Average_auto_calc = np.mean(Average_auto, axis=(0))

D = 1/3*np.trapz(Average_auto_calc)
print(D)

'''
