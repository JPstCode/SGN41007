import numpy as np
from matplotlib import pyplot as plt

#4

#a)

f0 = 0.017
mu, sigma = 0, 0.25 #mean and standard deviation
n = np.arange(100)

w = np.sqrt(0.25) * np.random.rand(100)
x = (np.sin(2 * np.pi * f0 * n) + w)

#plt.figure(0)
#plt.plot(x)
#plt.show()


#b)

scores = []
freq = []

for f in np.linspace(0, 0.5, 1000):

    #e = np.power(np.e,(-2*np.pi*f*x))
    n = np.arange(100)
    z = -2*np.pi*1j*f*n
    e = np.exp(z)
    score = np.dot(x,e)
    scores.append(score)
    freq.append(f)

scores = np.asarray(scores)
freq = np.asarray(freq)

fhat = freq[np.argmax(scores)]
print(fhat)
