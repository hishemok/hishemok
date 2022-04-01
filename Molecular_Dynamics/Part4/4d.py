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
a,v,r,N,L = placement(6,1.7)

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

def rdf(bin_edges, r , V):
    N = r.shape[0]

    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_sizes = bin_edges[1:] - bin_edges[:-1]
    n = np.zeros_like(bin_sizes)

    for i in range(N):
        dr = np.linalg.norm(r - r[i], axis=1)
        n += np.histogram(dr, bins=bin_edges)[0]
    
    n[0] = 0

    rdf = V / N**2 * n / (4 * np.pi * bin_centres**2 * bin_sizes) 
    
    return rdf


@njit
def func(i, a, v, r, L):

    r[:,i+1,:] = r[:,i,:] + v[:,i,:]*dt + 0.5*a[:,i,:]*dt**2
    r[:,i+1,:] = r[:,i+1,:] - np.floor(r[:,i+1,:] / L) * L

    a_new, u = calc_acc_pbc(r[:, i+1], 3, L)
   
    a[:, i+1] = a_new

    v[:,i+1,:] = v[:,i,:] + 0.5*(a[:,i,:] + a[:,i+1,:])*dt



a_new, u = calc_acc_pbc(r[:, 0], 3, L)
a[:, 0] = a_new


for i in trange(n-1):
    func(i, a, v, r, L)


cutoff = 3
num_bins = 51
rad_dis_list = np.linspace(0,cutoff,num_bins)
rad_dis = rdf(rad_dis_list,r[:,-1,:],L**3)


import matplotlib.pyplot as plt
plt.plot(rad_dis)
plt.ylabel('g(r)')
plt.xlabel('r')
plt.show()
