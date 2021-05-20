import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dense, AveragePooling2D,
                                     Flatten, BatchNormalization, LeakyReLU, Reshape)
import random
import matplotlib.pyplot as plt


def Denoise_model(shape):
    model = tf.keras.models.Sequential()

    #Decoder
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=shape, padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

    #Encoder
    model.add(Conv2DTranspose(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))

    model.add(Conv2D(1, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.compile(optimizer='adam', loss='MeanSquaredError', metrics=['accuracy'])

    return model


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


def plot_img(data):
    plt.imshow(data[:, :, 0], cmap='gray')
    plt.show()


solve_cudnn_error()

(x_train, a), (x_test, b) = tf.keras.datasets.mnist.load_data()
y_train = np.expand_dims(x_train, axis=3) / 255
y_test = np.expand_dims(x_test, axis=3) / 255

(x_noise_train, a), (x_noise_test, b) = tf.keras.datasets.mnist.load_data()
x_noise_train = add_noise(np.expand_dims(x_noise_train, axis=3), 0.3) / 255
x_noise_test = add_noise(np.expand_dims(x_noise_test, axis=3), 0.3) / 255

Denoise = Denoise_model(x_noise_train[0].shape)

fit_history_noise = Denoise.fit(x=x_noise_train, y=y_train, epochs=10, shuffle=True,
                                validation_split=0.1, verbose=1)

y_pred_test = Denoise.predict(x_noise_test)
print(y_pred_test.shape)

plot_img(y_pred_test[0])
plot_img(x_noise_test[0])


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


def add_noise(target, lv):
    noise_lv = lv
    img_size = 28 * 28

    for i in range(len(target)):
        ran_seq = random.sample([n for n in range(img_size)], np.int(img_size * noise_lv))
        x = target[i].reshape(-1, img_size)
        x[0, ran_seq] = 255

    return target


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
y_train = tf.one_hot(tf.squeeze(y_train), depth=10)
y_test = tf.one_hot(tf.squeeze(y_test), depth=10)

x_test_noise = add_noise(x_test, 0.3) / 255

CNN = CNN_model(x_train[0].shape, 10)

fit_history = CNN.fit(x=x_train, y=y_train, epochs=20, validation_split=0.1, verbose=1)
x_test = Denoise.predict(x_test_noise)

test_result = tf.keras.metrics.Accuracy()
y_pred = tf.round(CNN.predict(x_test))
test_result.update_state(y_pred, y_test)

print(test_result.result())