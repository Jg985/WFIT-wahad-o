import matplotlib.pyplot as plt
import numpy as np


f = open("tdot0.txt", "r")
pr=(f.read().split(","))
pr=list(map(float, pr))
t2 = np.linspace(0, 90, 90)
plt.plot(t2, pr, 'k', label='T/T0')
plt.legend(loc='best')
plt.grid()
plt.show()