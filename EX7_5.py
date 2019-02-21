from scipy.io import loadmat
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

data = loadmat('arcene.mat')

x_train = data['X_train']
x_test = data['X_test']
y_train = data['y_train'].ravel()
y_test = data['y_test'].ravel()

clf = LogisticRegression(penalty='l1',random_state=0)

C_range = []
for i in range(-10,10):
    C_range.append(float('10e'+str(i)))

C_range = np.asarray(C_range)


clf.fit(x_train,y_train)

for C in C_range:

    clf.C = C
    clf.fit(x_train, y_train)

    print('Number of selected features', np.count_nonzero(clf.coef_))
    predict = clf.predict(x_test)
    print('Accuracy: ', accuracy_score(y_test, predict))
    input('C value: ' + str(C))