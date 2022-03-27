import matplotlib.pyplot as plt
import numpy as np


epsilon = 1
sigma = 1.5
r = np.linspace(0.9,3,300)
U = lambda r,sigma:4*epsilon*((sigma/r)**(12)-(sigma/r)**(6))

#at first the potential exploded when around r<1.2
#therefor i changed r to start at 1.4 to get another view of the potential
#After that i changed r back to start at 0.9 and changed sigma to 0.95
plt.plot(r,U(r,sigma))
plt.xlabel('distance')
plt.ylabel('potential')
plt.show()
