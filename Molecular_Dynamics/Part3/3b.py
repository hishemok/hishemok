import numpy as np
import matplotlib.pyplot as plt

N = 4
n = 5000
dt = 0.001 
L = N
a = np.zeros((N,n,3))
v = np.zeros((N,n,3))
r = np.zeros((N,n,3))

r_norm_list = np.zeros((N,n))
       
r[0,0,:],r[1,0,:],r[2,0,:],r[3,0,:] = np.array([1,0,0]),np.array([0,1,0]),np.array([-1,0,0]),np.array([0,-1,0])



for i in range (n-1):
    for j in range(N):
        for l in range(j,N):
            R = r[j,i,:] - r[l,i,:]
            r_norm = np.linalg.norm(R)
            if r_norm < 3 and r_norm != 0:
                r_norm_list[j,i] = 4*(r_norm**(-12) - r_norm**(-6))
                r_norm_list[l,i] = r_norm_list[j,i]
                a[j,i,:] += 24*( 2*(r_norm**(-12)) - (r_norm**(-6)) ) * (R)/r_norm**2
                
                a[l,i,:] -= 24*( 2*(r_norm**(-12)) - (r_norm**(-6)) ) * (R)/r_norm**2
       

    r[:,i+1,:] = r[:,i,:] + v[:,i,:]*dt + 0.5*a[:,i,:]*dt**2
    v[:,i+1,:] = v[:,i,:] + 0.5*(a[:,i,:] + a[:,i+1,:])*dt


def K(v):
    return 0.5*np.sum(v**2,axis=(0,2))

def U(r_norm_list):
    sum = np.zeros(n)
    for i in range(N):
        for j in range(n):
            sum[j] += r_norm_list[i,j]
    sum[-1] = sum[-2]
    return sum

t = np.linspace(0,dt*n,n)

plt.plot(t,U(r_norm_list),label='Potential')
plt.plot(t,K(v),label='Kinetic')
plt.plot(t,K(v)+U(r_norm_list),label='Total')
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

