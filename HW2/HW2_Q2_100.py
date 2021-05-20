import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras.layers import Dense


def one_hot(features):
    feature_dict = {}
    i = 0
    for row in features:
        for element in row.split('; '):
            if element not in feature_dict:
                feature_dict[element] = i
                i = i+1

    feature_list = []
    for rows in range(len(features)):
        temp = [0]*(i)
        for feature_str in feature_dict:
            if feature_str in features[rows]:
                temp[feature_dict[feature_str]] = 1
        feature_list.append(temp)

    return feature_list, list(feature_dict.keys())


def label_y(features):
    label = []
    for row in range(len(features)):
        if features[row] == 'Access Control':
            label.append(0)
        else:
            label.append(1)
    return np.asarray(label)


def load_data(data_path, feature_name):
    with open(data_path, newline='') as filename:
        rows = csv.DictReader(filename)

        lines = []
        for row in rows:
            lines.append(row[feature_name])

    return lines


def dense_model(feature_columns):
    model = tf.keras.Sequential()

    model.add(Dense(64, activation='tanh', input_shape=(feature_columns,)))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=tf.keras.optimizers.Adam(0.005))

    return model


def combine_array(A, B):
    for row in range(len(A)):
        A[row] = A[row]+B[row]
    return np.asarray(A)


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

path = 'train_DefenseSystem.csv'
event_rule_name, name_list = one_hot(load_data(path, 'event_rule_name'))
event_rule_reference, reference_list = one_hot(load_data(path, 'event_rule_reference'))
x_train = combine_array(event_rule_name, event_rule_reference)
y_train = label_y(load_data(path, 'event_rule_category'))
print(x_train.shape)
print(name_list)

classifier = dense_model(x_train.shape[1])
train_history = classifier.fit(x=x_train, y=y_train, validation_split=0.2,
                               epochs=30, verbose=2)
