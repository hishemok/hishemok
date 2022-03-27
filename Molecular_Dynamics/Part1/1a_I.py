import matplotlib.pyplot as plt
import numpy as np

epsilon = 1
sigma = 1
r = np.linspace(0.9,3,300)
U = lambda r,sigma:4*epsilon*((sigma/r)**(12)-(sigma/r)**(6))

plt.plot(r,U(r,sigma))
plt.show()
