import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Dense, BatchNormalization)
import scipy.io as sio


def get_data(file_name):
    data = sio.loadmat(file_name)
    x = tf.convert_to_tensor(np.hstack([data['x1'], data['x2']]))
    y = tf.convert_to_tensor(data['y'])

    return x, y


def dense_model():
    model = tf.keras.Sequential()

    model.add(Dense(3, activation='tanh', input_shape=(2,)))
    model.add(Dense(2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.005))

    return model


classifier = dense_model()

x_train, y_train = get_data('train.mat')
x_test, y_test = get_data('test.mat')

train_history = classifier.fit(x=x_train, y=y_train, validation_split=0.1,
                               epochs=250, verbose=2)

test_result = tf.keras.metrics.Accuracy()
y_pred = classifier.predict_classes(x_test)
test_result.update_state(y_pred, y_test)

print(test_result.result())