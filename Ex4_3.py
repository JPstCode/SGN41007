import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

digits = load_digits()

print(digits.data.shape)
print(digits.target.shape)

xtrain, xtest, ytrain, ytest = train_test_split(digits.data,digits.target, test_size= 0.2)


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
for classifier in classifiers:
    classifier.fit(xtrain,ytrain)
    pred = classifier.predict(xtest)
    print(cl_names[index], accuracy_score(ytest,pred))
    index+=1



