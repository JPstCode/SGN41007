import numpy as np
from matplotlib import pyplot as plt

xtest = np.load("X_test_kaggle.npy")
xtrain = np.load("X_train_kaggle.npy")
y_train = []


with open('y_train_final_kaggle.csv') as fp:
    for line in fp:
        class_ = line.split(',')
        y_train.append(class_[1].rstrip())

y_train = np.asarray(y_train)


