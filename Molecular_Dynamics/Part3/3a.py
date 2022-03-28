import numpy as np

N = 4
n = 500
dt = 0.01 

a = np.zeros((N,n,3))
v = np.zeros((N,n,3))
r = np.zeros((N,n,3))

r_norm_list = np.zeros((N,n))
 
#giving starting positions so that the particles do not all have the same startingposition
r[0,0,:],r[1,0,:],r[2,0,:],r[3,0,:] = np.array([1,0,0]),np.array([0,1,0]),np.array([-1,0,0]),np.array([0,-1,0])


for i in range (n-1):
    for j in range(N):
        for l in range(j,N):
            R = r[j,i,:] - r[l,i,:]
            r_norm = np.linalg.norm(R)
            #r_norm = r_norm - np.round(r_norm/L)*L
            if r_norm < 3 and r_norm != 0:
                r_norm_list[j,i] = 4*(r_norm**(-12) - r_norm**(-6))
                r_norm_list[l,i] = r_norm_list[j,i]
                a[j,i,:] += 24*( 2*(r_norm**(-12)) - (r_norm**(-6)) ) * (R)/r_norm**2
                
                a[l,i,:] -= 24*( 2*(r_norm**(-12)) - (r_norm**(-6)) ) * (R)/r_norm**2
       

    r[:,i+1,:] = r[:,i,:] + v[:,i,:]*dt - 0.5*a[:,i,:]*dt**2
    v[:,i+1,:] = v[:,i,:] + 0.5*(a[:,i,:] + a[:,i+1,:])*dt


