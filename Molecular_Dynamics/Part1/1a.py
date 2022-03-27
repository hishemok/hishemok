import numpy as np
import matplotlib . pyplot as plt
epsilon , sigma = 1 ,1
n = 1000
r = np.linspace(0.9,3,n+1) 

def u(r,epsilon ,sigma):
  u = 4∗epsilon∗( (sigma/r)∗∗12 − (sigma/r)∗∗6 ) 
  return u
 
U = u(r,epsilon ,sigma)
  
plt.plot( r , U )
plt.xlabel( ' r ' ) 
plt.ylabel( 'u( r ) ' )
plt.show()
