import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


X = np.load('gtsrb_x.npy')
y = np.load('gtsrb_y.npy')
f = np.load('gtsrb_f.npy')

neigh = KNeighborsClassifier()
LDA = LinearDiscriminantAnalysis()
svm = SVC()
LR = LogisticRegression()

classifiers = []
cl_names = ["knn", "LDA", "svm", "LR"]

classifiers.append(neigh)
classifiers.append(LDA)
classifiers.append(svm)
classifiers.append(LR)

index = 0
for clfier in classifiers:

    print(cl_names[index], cross_val_score(clfier, f, y))
    index+=1
