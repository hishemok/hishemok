import numpy as np

n = 500
dt = 0.01
sigma = 1.5

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

a,v,r,r_d,t = Velocity_verlet(sigma)

f = open('Oppg2d.txt','w')
for i in range(n):
    f.write('2 \n')
    f.write('Type   x     y      z \n')
    f.write(f'Ar   {r[0,i,0]}     {r[0,i,1]}     0 \n')
    f.write(f'Ar   {r[1,i,0]}     {r[1,i,1]}     0 \n')
f.close()
