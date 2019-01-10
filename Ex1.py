#First assignment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#Part 1
fig = plt.figure()
X = []

csv = np.loadtxt("EX1/locationData.csv")
print(csv.shape)


#Part 2

#plt.plot(csv[0],csv[1])
ax = fig.add_subplot(111, projection="3d")
plt.plot(csv[0],csv[1],csv[2])

#plt.show()


#Part 3

with open('EX1/locationData.csv') as fp:
    for line in fp:
        values = line.split(" ")
        values = [float(v) for v in values]
        X.append(values)


X = np.asarray(X)

