import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


c1 = np.load('GTSRB/class1.npy')
c2 = np.load('GTSRB/class2.npy')


c1 = c1.reshape([len(c1),64,-1])
c2 = c2.reshape([len(c2),64,-1])

data = []
labels = []
for img in c1:
    data.append(normalize(img))
    labels.append(0)

for img in c2:
    data.append(normalize(img))
    labels.append(1)

data = np.asarray(data).reshape([-1,64,64,3])
labels = np.asarray(labels)


xtrain, xtest, ytrain, ytest = train_test_split(data,labels,test_size=0.2)