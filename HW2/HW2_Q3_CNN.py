import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (Conv2D, AveragePooling2D, Flatten, Dense)
import random

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
print(x_train.shape)
y_train = tf.one_hot(tf.squeeze(y_train), depth=10)
y_test = tf.one_hot(tf.squeeze(y_test), depth=10)


def CNN_model(shape, classes):
    model = tf.keras.models.Sequential()

    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                     input_shape=shape, padding='same'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                     padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Flatten())

    model.add(Dense(120, activation='tanh'))
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


CNN = CNN_model(x_train[0].shape, 10)

fit_history = CNN.fit(x=x_train, y=y_train, epochs=20, validation_split=0.1, verbose=1)

test_result = tf.keras.metrics.Accuracy()
y_pred = tf.round(CNN.predict(x_test))
test_result.update_state(y_pred, y_test)

print(test_result.result())


def add_noise(target, lv):
    noise_lv = lv
    img_size = 28 * 28

    for i in range(len(target)):
        ran_seq = random.sample([n for n in range(img_size)], np.int(img_size * noise_lv))
        x = target[i].reshape(-1, img_size)
        x[0, ran_seq] = 255

    return target


def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


solve_cudnn_error()

x_test_noise = add_noise(x_test, 0.3)

test_result_2 = tf.keras.metrics.Accuracy()
y_pred = tf.round(CNN.predict(x_test_noise))
test_result_2.update_state(y_pred, y_test)

print(test_result_2.result())


x_train_noise = add_noise(x_train, 0.3)

CNN_noise_model = CNN_model(x_train_noise[0].shape, 10)

fit_history_noise = CNN_noise_model.fit(x=x_train_noise, y=y_train, epochs=20,
                                        validation_split=0.1, verbose=1)

test_result_3 = tf.keras.metrics.Accuracy()
y_pred_noisy = tf.round(CNN_noise_model.predict(x_test))
test_result_3.update_state(y_pred_noisy, y_test)

print(test_result_3.result())

test_result_4 = tf.keras.metrics.Accuracy()
y_pred_noisy2 = tf.round(CNN_noise_model.predict(x_test_noise))
test_result_4.update_state(y_pred_noisy2, y_test)

print(test_result_4.result())
