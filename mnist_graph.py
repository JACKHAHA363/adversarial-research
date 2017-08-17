import tensorflow as tf
import keras.backend as K
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Flatten ,Dropout
from keras.models import Sequential

def get_model(model_name):
    model_dict = {
        'mlp' : mlp,
        'conv' : conv,
        'conv2' : conv2,
        'linear' : linear,
        'A' : modelA,
        'B' : modelB,
        'C' : modelC,
        'D' : modelD,
        'conv2_rep': conv2_rep,
    }
    try:
        return model_dict[model_name]((28,28,1))
    except KeyError:
        print "{} is not an available model".format(model_name)
        exit()

def modelA(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(10, activation=None))
    return model

def modelB(input_shape):
    model = Sequential()
    model.add(Dropout(0.2, input_shape=input_shape))
    model.add(Conv2D(64, (8, 8),
                            strides=(2, 2),
                            padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (6, 6),
                            strides=(2, 2),
                            padding='valid'))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (5, 5),
                            strides=(1, 1)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(10, activation=None))
    return model

def modelC(input_shape):
    model = Sequential()
    model.add(Conv2D(128, (3, 3),
                            padding='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(10, activation=None))
    return model

def modelD(input_shape):
    model = Sequential()

    model.add(Flatten(input_shape=input_shape))

    model.add(Dense(300, kernel_initializer='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, kernel_initializer='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, kernel_initializer='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, kernel_initializer='he_normal', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation=None))
    return model

def linear(input_shape):
    net = Sequential()
    net.add(Flatten(input_shape=input_shape))
    net.add(Dense(10, activation=None))
    return net

def mlp(input_shape):
    net = Sequential()
    net.add(Flatten(input_shape=input_shape))
    net.add(Dense(512, activation='relu'))
    net.add(Dropout(0.2))
    net.add(Dense(512, activation='relu'))
    net.add(Dropout(0.2))
    net.add(Dense(10, activation=None))
    return net

def conv(input_shape):
    net = Sequential()
    net.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=input_shape))
    net.add(Conv2D(64, (3, 3), activation='relu'))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Dropout(0.25))
    net.add(Flatten())
    net.add(Dense(128, activation='relu'))
    net.add(Dropout(0.5))
    net.add(Dense(10, activation=None))
    return net

def conv2(input_shape):
    net = Sequential()
    net.add(
        Conv2D(
            filters=32, kernel_size=(3,3), strides=(1,1),
            padding='valid', data_format='channels_last',
            activation="relu", input_shape=input_shape,
        )
    )
    net.add(
            Conv2D(
            filters=32, kernel_size=(3,3), strides=(1,1),
            padding='valid', data_format='channels_last',
            activation="relu", input_shape=input_shape,
        )
    )
    net.add(
        MaxPooling2D(pool_size=(2,2), data_format="channels_last")
    )

    net.add(
        Conv2D(
            filters=64, kernel_size=(3,3),
            strides=(1,1), padding='valid',
            data_format='channels_last', activation="relu"
        )
    )
    net.add(
        Conv2D(
            filters=64, kernel_size=(3,3),
            strides=(1,1), padding='valid',
            data_format='channels_last', activation="relu"
        )
    )

    net.add(
        MaxPooling2D(pool_size=(2,2), data_format="channels_last")
    )

    net.add(Flatten())

    net.add(Dense(units=200, activation='relu'))
    net.add(Dropout(rate=0.5))

    net.add(Dense(units=200, activation='relu'))
    net.add(Dropout(rate=0.5))

    net.add(Dense(units=10, activation=None))
    return net

def conv2_rep(input_shape):
    net = Sequential()
    net.add(
        Conv2D(
            filters=32, kernel_size=(3,3), strides=(1,1),
            padding='valid', data_format='channels_last',
            activation="relu", input_shape=input_shape,
        )
    )
    net.add(
            Conv2D(
            filters=32, kernel_size=(3,3), strides=(1,1),
            padding='valid', data_format='channels_last',
            activation="relu", input_shape=input_shape,
        )
    )
    net.add(
        MaxPooling2D(pool_size=(2,2), data_format="channels_last")
    )

    net.add(
        Conv2D(
            filters=64, kernel_size=(3,3),
            strides=(1,1), padding='valid',
            data_format='channels_last', activation="relu"
        )
    )
    net.add(
        Conv2D(
            filters=64, kernel_size=(3,3),
            strides=(1,1), padding='valid',
            data_format='channels_last', activation="relu"
        )
    )

    net.add(
        MaxPooling2D(pool_size=(2,2), data_format="channels_last")
    )

    net.add(Flatten())

    net.add(Dense(units=200, activation='relu'))
    net.add(Dropout(rate=0.5))

    net.add(Dense(units=200, activation='relu'))
    net.add(Dropout(rate=0.5))

    net.add(Dense(units=10, activation=None))
    return net

