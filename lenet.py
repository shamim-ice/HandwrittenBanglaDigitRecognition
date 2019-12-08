from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation
from keras import backend as K


class LeNet:
    @staticmethod
    def build(height, width, depth, classes):
        inputShape = (height, width, depth)

        if K.image_data_format() == 'channel_first':
            inputShape = (depth, height, width)

        model = Sequential()
        model.add(Conv2D(32, (5, 5), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(1000))
        model.add(Activation('relu'))

        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model
