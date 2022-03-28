import numpy as np
import matplotlib.pyplot as plt

n = 500
dt = 0.01


def Euler(sigma):
    time = np.linspace(0,n*dt,n)
    v = np.zeros((2,n,2))
    r = np.zeros(v.shape)
    a = np.zeros(v.shape)
    r_distance = np.zeros(n)

    r[0,0,:],r[0,0,:] = np.array([0,0]),np.array([0,sigma])
    r_distance[0] = np.linalg.norm(r[0,0,:]-r[1,0,:])

    for i in range(n-1):
        r_norm = np.linalg.norm(r[0,i,:]-r[1,i,:])
        norm_sqr = r_norm**2
        r_distance[i+1] = r_norm

        a[0,i,:] = 24*( 2*(norm_sqr**(-6)) - (norm_sqr**(-3)) )*(r[0,i,:]-r[1,i,:])/norm_sqr
        a[1,i,:] = - a[0,i,:]

        v[:,i+1,:] = v[:,i,:] + a[:,i,:]*dt
        r[:,i+1,:] = r[:,i,:] + v[:,i,:]*dt
    return a,v,r,r_distance,time




def Euler_Cromer(sigma):
    time = np.linspace(0,n*dt,n)
    v = np.zeros((2,n,2))
    r = np.zeros(v.shape)
    a = np.zeros(v.shape)
    r_distance = np.zeros(n)

    r[0,0,:],r[0,0,:] = np.array([0,0]),np.array([0,sigma])
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



def Velocity_verlet(sigma):
    time = np.linspace(0,n*dt,n)
    v = np.zeros((2,n,2))
    r = np.zeros(v.shape)
    a = np.zeros(v.shape)
    r_distance = np.zeros(n)

    r[0,0,:],r[0,0,:] = np.array([0,0]),np.array([0,sigma])
    r_distance[0] = np.linalg.norm(r[0,0,:]-r[1,0,:])

    for i in range(n-1):
        r_norm = np.linalg.norm(r[0,i,:]-r[1,i,:])
        norm_sqr = r_norm**2
        r_distance[i+1] = r_norm

        a[0,i,:] = 24*( 2*(norm_sqr**(-6)) - (norm_sqr**(-3)) )*(r[0,i,:]-r[1,i,:])/norm_sqr
        a[1,i,:] = - a[0,i,:]

        r[:,i+1,:] = r[:,i,:] + v[:,i,:]*dt + 0.5*a[:,i,:]*dt**2
        v[:,i+1,:] = v[:,i,:] + 0.5*(a[:,i,:]+a[:,i+1,:])*dt
        
    return a,v,r,r_distance,time


def U(r):
    return 4*((r)**(-12)-(r)**(-6))

def K(v):
    return 0.5*np.sum(v**2,axis=(0,2))

#All different plots came from changing sigma and what function i used
  
sigma = 1.5
a,v,r,r_d,t = Euler_Cromer(sigma)


plt.plot(t,U(r_d),label='Potential')
plt.plot(t,K(v),label='Kinetic')
plt.plot(t,K(v)+U(r_d),label='Total')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.show()
