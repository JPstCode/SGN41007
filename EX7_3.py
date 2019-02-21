import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential,Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from GTSRB_prep import xtrain, xtest, ytrain, ytest


vgg_model = VGG16(weights='imagenet',
                  include_top='False')


model = Sequential()
model.add(VGG16())
model.summary()


#layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
#x = layer_dict['block2_pool'].output

#x = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(x)
#x = MaxPooling2D(pool_size=(2,2))(x)











