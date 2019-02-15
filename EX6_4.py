from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from EX6_3 import xtrain, xtest, ytrain, ytest
import numpy as np

N = 10
w, h = 5,5
epochs = 20

ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)

model = Sequential()

model.add(Conv2D(32,(5,5),
                 input_shape=(64,64,3),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(4,4)))

model.add(Conv2D(32,(5,5),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(4,4))

model.add(Flatten())
model.add(Dense(100,activation='sigmoid'))
model.add(Dense(2,activation='sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics = ['accuracy'])

model.fit(xtrain,ytrain,epochs,
          validation_data=(xtest,ytest))
