from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np

N = 10
w, h = 5,5

model = Sequential()

model.add(Conv2D(N,(w,h),
                 input_shape=(64,64,3),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(N,(w,h),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(2,activation='sigmoid'))


model.summary()