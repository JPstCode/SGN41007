import numpy as np
from matplotlib import pyplot as plt

def load_data():
    xtest = np.load("X_test_kaggle.npy")
    xtrain = np.load("X_train_kaggle.npy")
    y_train = []

    with open('y_train_final_kaggle.csv') as fp:
        for line in fp:
            class_ = line.split(',')
            y_train.append(class_[1].rstrip())

    y_train.pop(0)
    y_train = np.asarray(y_train)

    return xtrain, y_train, xtest

def sorting_data(xtrain,ytrain):

    sorted_data = []
    index = 0
    for element in ytrain:

        if element == "tiled":
            print("tiled")

        elif element =='soft_tiles':
            print('soft_tiles')

        elif element =='hard_tiles':
            print('hard_tiles')

        elif element =='hard_tiles_large_space':
            print('hard_tiles_large')

        elif element =='fine_concrete':
            print('fine_conc')

        elif element =='concrete':
            print('conc')

        elif element =='wood':
            print('wood')

        elif element =='soft_pvc':
            print('soft_pct')


        elif element =='carpet':
            print('carpet')


        index+=1
        input(index)




def main():

    xtrain, y_train, xtest = load_data()
    sorting_data(xtrain, y_train)


main()