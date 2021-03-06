import numpy as np
import matplotlib.pyplot as plt

def Euler_Cromer():
    time = np.linspace(0,n*dt,n)
    v = np.zeros((2,n,2))
    r = np.zeros(v.shape)
    a = np.zeros(v.shape)
    r_distance = np.zeros(n)

    r[0,0,:],r[0,0,:] = np.array([0,0]),np.array([0,1.5])
    r_distance[0] = np.linalg.norm(r[0,0,:]-r[1,0,:])

    for i in range(n-1):
        r_norm = np.linalg.norm(r[0,i,:]-r[1,i,:])
        norm_sqr = r_norm**2
        r_distance[i+1] = r_norm

        a[0,i,:] = 24*( 2*(norm_sqr**(-6)) - (norm_sqr**(-3)) )*(r[0,i,:]-r[1,i,:])/norm_sqr
        a[1,i,:] = - a[0,i,:]

        v[:,i+1,:] = v[:,i,:] + a[:,i,:]*dt
        r[:,i+1,:] = r[:,i,:] + v[:,i+1,:]*dt
    return a,v,r,r_distance,time

a,v,r,r_d,t = Euler_Cromer()
plt.plot(t,r_d)
plt.show()
