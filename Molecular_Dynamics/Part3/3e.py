import numpy as np

N = 1
L = 2
a = np.zeros((N,n,3))
v = np.zeros(a.shape)
r = np.zeros(a.shape)
r[0,0,:] = np.array([1,0,0])
v[0,0,:] = r[0,0,0]



@njit
def calc_acc_pbc(r, cutoff,L):
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


U = np.zeros(n)
def func(i, a, v, r,L):

    r[:,i+1,:] = r[:,i,:] + v[:,i,:]*dt + 0.5*a[:,i,:]*dt**2
    r[:,i+1,:] = r[:,i+1,:] - np.floor(r[:,i+1,:] / L) * L

    a_new, u = calc_acc_pbc(r[:, i+1], 3,L)
    U[i+1] = u
   
    a[:, i+1] = a_new

    v[:,i+1,:] = v[:,i,:] + 0.5*(a[:,i,:] + a[:,i+1,:])*dt

   


a_new, u = calc_acc_pbc(r[:, 0], 3,L)
a[:, 0] = a_new

for i in trange(n-1):
    func(i, a, v, r,L)

    
    
file = open('Project.txt','w')
for i in range(n):

    file.write(f'{N} \n')
    file.write("Type            x                y              z \n")
    for j in range(N):  
        file.write(f'Ar    {r[j,i,0]}     {r[j,i,1]}      {r[j,i,2]}   \n')
       
file.close()


