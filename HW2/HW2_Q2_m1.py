import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout


def one_hot(features):
    feature_dict = {}
    for element in features:
        if element not in feature_dict:
            feature_dict[element] = 1
        elif element in feature_dict:
            feature_dict[element] = feature_dict[element] + 1
    if len(feature_dict) > 50:
        feature_dict = check_frequency(feature_dict)

    feature_list = []
    for item in feature_dict.keys():
        feature_list.append(item)

    ret_list = []
    for rows in range(len(features)):
        temp = [0]*(len(feature_list))
        for item in range(len(feature_list)):
            if features[rows] == feature_list[item]:
                temp[item] = 1
        ret_list.append(temp)

    return ret_list, feature_list


def check_frequency(feature_dict):
    del_list = []
    for element in feature_dict:
        if feature_dict[element] < 2:
            del_list.append(element)

    for element in del_list:
        del feature_dict[element]

    return feature_dict


def load_x(data_path):
    extract_feature = ['device_dev_name', 'device_hashed_mac', 'event_protocol_id',
                       'event_flow_outbound_or_inbound', 'event_role_device_or_router',
                       'event_role_server_or_client', 'event_rule_severity']
    features = {}
    feature_list = []
    for item in extract_feature:
        with open(data_path, newline='') as filename:
            rows = csv.DictReader(filename)
            temp = []
            for row in rows:
                temp.append(row[item])
        temp, feature_list_temp = one_hot(temp)
        feature_list += feature_list_temp
        features[item] = temp

    lines = []
    for rows in range(len(features[extract_feature[0]])):
        temp = []
        for item in extract_feature:
            temp = temp + features[item][rows]
        lines.append(temp)

    return np.asarray(lines), feature_list


def load_data(data_path, feature_name):
    with open(data_path, newline='') as filename:
        rows = csv.DictReader(filename)

        lines = []
        for row in rows:
            lines.append(row[feature_name])

    return lines


def label_y(features):
    label = []
    for row in range(len(features)):
        if features[row] == 'Access Control':
            label.append(0)
        else:
            label.append(1)
    return np.asarray(label)


def dense_model(feature_columns):
    model = tf.keras.Sequential()

    model.add(Dense(32, activation='relu', activity_regularizer=tf.keras.regularizers.l1(0.0001), input_shape=(feature_columns,)))
    model.add(Dropout(0.45))
    model.add(Dense(16, activation='relu', activity_regularizer=tf.keras.regularizers.l1(0.0001)))

    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=tf.keras.optimizers.Adam(0.005))

    return model


def load_test_data(data_path, target_features):
    extract_feature = ['device_dev_name', 'device_hashed_mac', 'event_protocol_id',
                       'event_flow_outbound_or_inbound', 'event_role_device_or_router',
                       'event_role_server_or_client', 'event_rule_severity']

    feature_list = []
    with open(data_path, newline='') as filename:
        rows = csv.DictReader(filename)

        for row in rows:
            temp = []
            for item in extract_feature:
                temp.append(row[item])
            feature_list.append(temp)

    lines = []
    for rows in feature_list:
        temp = [0]*(len(target_features))
        for feature in rows:
            if feature in target_features:
                temp[target_features.index(feature)] = 1
        lines.append(temp)

    return np.asarray(lines)


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
x_train, features = load_x(path)
y_train = label_y(load_data(path, 'event_rule_category'))
print(x_train.shape)

classifier = dense_model(x_train.shape[1])
train_history = classifier.fit(x=x_train, y=y_train, validation_split=0.2,
                               epochs=30, verbose=2)

test_path = 'test_DefenseSystem.csv'
x_test = load_test_data(test_path, features)
y_pred = classifier.predict_classes(x_test)

with open('result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['event_rule_category'])
    for row in y_pred:
        if row == [0]:
            writer.writerow(['Access Control'])
        else:
            writer.writerow(['Web Attack'])
