import numpy as np
from matplotlib import pyplot as plt

y = np.zeros(500)
s2 = np.zeros(300)
n = np.arange(0,100)
signal = np.cos(2 * np.pi * 0.03 * n)

y = np.concatenate((y, signal), axis=None)
y = np.concatenate((y, s2), axis=None)

#a)

plt.subplot(4,1,1)
plt.plot(y)

#b)


y_n = y + np.sqrt(0.5)*np.random.randn(y.size)

plt.subplot(4,1,2)
plt.plot(y_n)


#c)