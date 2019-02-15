from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from EX6_3 import xtrain, xtest, ytrain, ytest
import numpy as np

#loss = categorical crossentropy
#optimiser = stochastic gradient descent

Minibatch = 32
epochs = 20
w,h = 5,5

ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)

model = Sequential()

model.add(Conv2D(10,(w,h), input_shape=(64,64,3),
                 activation='relu'))
model.add(Conv2D(20,(w,h), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(10,(w,h), input_shape=(32,32,20),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(10, activation='relu'))

#number of classes 2
model.add(Dense(2,activation='relu'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics = ['accuracy'])

model.fit(xtrain,ytrain,epochs,
          validation_data=(xtest,ytest))


