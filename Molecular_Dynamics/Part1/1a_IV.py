



#up until this point the code is from 1a_I.py

U_15 = u(r,epsilon,1.5)
U_095 = u(r,epsilon,0.95)
plt.subplot(2,1,1)
plt.plot(r,U_15,label='sigma=1.5')
plt.xlabel('r')
plt.ylabel('u(r)')
plt.legend()
plt.subplot(2,1,2)
plt.plot(r,U_095,label='sigma=0.95')
plt.xlabel('r')
plt.ylabel('u(r)')
plt.legend()
plt.show()
