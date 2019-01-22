import numpy as np
from matplotlib import pyplot as plt

y = np.zeros(500)
s2 = np.zeros(300)
n = np.arange(0,100)
signal = np.cos(2 * np.pi * 0.1 * n)

y = np.concatenate((y, signal), axis=None)
y = np.concatenate((y, s2), axis=None)

#a)

plt.subplot(4,1,1)
plt.plot(y)

#b)

y_n = y + np.sqrt(0.5)*np.random.randn(y.size)

plt.subplot(4,1,2)
plt.plot(y_n)


#Convolve##

#n2 = np.arange(900)
#h = np.cos(2*np.pi*0.1*n2)

#asd = np.convolve(y_n,y,'same')

#plt.subplot(4,1,3)
#plt.plot(asd)
#plt.show()
#####################

sigma = 0.5
lh_ratios = []
detection_values = []
#result = []

for step in range(0,9):

    #100 unit long window
    n_w = np.arange(step*100,(step+1)*100)

    #Likelyhood-ratio
    lh_ratio = np.exp(-(1/2*sigma**2)*(np.sum((y_n[n_w]-np.cos(2*np.pi*0.1*n_w))**2)-np.sum(y_n[n_w]**2)))
    lh_ratios.append(lh_ratio)

    #Detector
    detector = np.sum(y_n[n_w]*np.cos(2*np.pi*0.1*n_w))
    detection_values.append(detector)

    #result.append(detector*signal)

detection_values = np.asarray(detection_values)
lh_ratios = np.asarray(lh_ratios)

plt.subplot(4,1,3)
plt.plot(detection_values)

#plt.subplot(4,1,4)
#plt.plot(lh_ratios)
#plt.show()