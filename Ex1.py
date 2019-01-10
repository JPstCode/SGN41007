#First assignment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Part 1
fig = plt.figure()
X = []

csv = np.loadtxt("EX1/locationData.csv")
#print(csv.shape)


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


#Part 4

mat = loadmat('EX1/twoClassData.mat')
#print(mat.keys())

xmat = mat['X']
y = mat['y'].ravel()

zeros = xmat[y == 0, :]
ones = xmat[y == 1, :]

#print(xmat[0:4])
#print((xmat[:,1][0:4]))
#print(xmat[:,0][0:4])
#print(zeros[0:4])
#print(ones[0:4])

plt.figure(2)
#plt.plot(xmat[:,0],xmat[:,1], 'ro' )
plt.plot(zeros[:,0], zeros[:,1], 'ro')
plt.plot(ones[:,0], ones[:,1], 'bo')
#plt.show()

#prt 5

xs = np.load('EX1/x.npy')
ys = np.load('EX1/y.npy')

#line => y = mx + c ==>> y = Ap (A = [[x 1]], p = [[m],[c]]

A = np.vstack([xs, np.ones(len(xs))]).T

m, c = np.linalg.lstsq(A, ys)[0]

plt.figure(3)
plt.plot(xs,ys, 'o', label='original data', markersize=5)
plt.plot(xs, m*xs + c, 'r', label='fittedline')
plt.show()