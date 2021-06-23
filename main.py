import numpy as np
from matplotlib import animation
from scipy.integrate import odeint
from numpy import sin, cos, pi, array
import matplotlib.pyplot as plt
from scipy.fft import fft,fftfreq,fftshift
var=16
#równanie wahadła rzeczywistego
def func (z, t,g):
    x,y=z
    return y,-g*sin(x)
z0 = [np.pi/3, 0.0]

g=10
t = np.linspace(0, 5, 2**var)
pend = odeint(func, z0, t, (g,))

#równanie wahadła matematycznego

def func1 (x1, t1,g1):
    return x1*sin(g1**(1/2)*t1+pi/2)

x1=pi/3
g1=10
t1 = np.linspace(0, 5, 2**var)
pend1=np.array(func1(x1,t1,g1))

#okres wahadła matematycznego (stały)

T=2*pi*(1/10)**(0.5)

#rysowanie wykresów

plt.plot(t, pend[:, 0], 'k', label='wahadło rzeczywiste')
plt.plot(t1, pend1, 'b', label='wahadło matematyczne')
plt.legend(loc='best')
plt.grid()
plt.show()

#fft wahadła rzeczywistego (dla kąta pi/3)

m=fft(pend[:,0]*np.hanning(2**var))
t = np.linspace(0, 5000, 2**var)

n=fftshift(abs(m))

mx=np.argmax(n)
print(mx)
freq=fftshift(fftfreq(2**var,t1[1]-t1[0]))
print(1/abs(freq[mx]))
i=1

#obliczanie okresu whadała rzeczywistego dla kątów 1-90

okr=[]
por=[]

while i<91:
    z0 = [np.radians(i), 0.0]
    pend = odeint(func, z0, t, (g,))
    m = fft(pend[:, 0] * np.hanning(2 ** var))
    n = fftshift(abs(m))
    mx = np.argmax(n)
    okr.append(1 / abs(freq[mx]))
    por.append(okr[i-1]/T)
    i=i+1
    print(i)

t2 = np.linspace(0, 90, 90)
plt.plot(t2, por, 'k', label='T/T0')
plt.legend(loc='best')
plt.grid()
plt.show()

print(por)
