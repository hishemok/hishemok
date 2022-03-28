import numpy as np

def placement(n_,L):

    N = 4*n_**3
    d = L/n_
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

a,v,r,N,L = placement(3,20)

file = open('Project.txt','w')
for i in range(n):

    file.write(f'{N} \n')
    file.write("Type            x                y              z \n")
    for j in range(N):  
        file.write(f'Ar    {r[j,i,0]}     {r[j,i,1]}      {r[j,i,2]}   \n')
       
file.close()
