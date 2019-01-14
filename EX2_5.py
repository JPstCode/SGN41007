import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np

# Read the data

img = imread("EX2/uneven_illumination.jpg")
plt.imshow(img, cmap='gray')
plt.title("Image shape is %dx%d" % (img.shape[1], img.shape[0]))
plt.show()

# Create the X-Y coordinate pairs in a matrix
#(1030, 1300)
X, Y = np.meshgrid(range(1300), range(1030))

Z = img

#(1339000,)
x = X.ravel()
y = Y.ravel()
z = Z.ravel()


#print(x[1299])
#print(y[1300])
#print(z.shape)
#input("asd")


# ********* TODO 1 **********
# Create data matrix
# Use function "np.column_stack".
# Function "np.ones_like" creates a vector like the input.

d1 = x**2
d2 = y**2
d3 = x*y

H = np.stack((d1,d2,d3,x,y,np.ones_like(x)))

# ********* TODO 2 **********
# Solve coefficients
# Use np.linalg.lstsq
# Put coefficients to variable "theta" which we use below.

data = H.T
theta = np.linalg.lstsq(data,z)[0]


# Predict


z_pred = np.dot(H.T,theta)
Z_pred = np.reshape(z_pred, X.shape)

# Subtract & show
S = Z - Z_pred
plt.figure(2)
plt.imshow(S, cmap = 'gray')
plt.show()
