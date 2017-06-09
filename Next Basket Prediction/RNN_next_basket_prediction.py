import json
import numpy as np
from numpy import random
from keras.models import Sequential
from keras_diagram import ascii
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, TimeDistributed
from keras.models import load_model

data = list()
with open('tafeng_json_data_more_transactions') as f:
    for line in f:
        data.append(sum(list(json.loads(line).values()[0]), []))

all_item_id = sum(data, [])

print(len(list(set(all_item_id))))
# print()

unique_item_id = list(set(all_item_id))


def get_unique_customer_id(filename):
    all_customer_id = list()
    with open(filename) as f:
        for line in f:
            all_customer_id.append(json.loads(line).keys()[0])
    unique_customer_id = list(set(all_customer_id))
    return unique_customer_id


def u_c_id_for_train_test(ratio):
    u_c_id = get_unique_customer_id('tafeng_json_data_more_transactions_item_id')
    random.shuffle(u_c_id)
    u_c_id_train = u_c_id[:int(ratio * len(u_c_id))]
    u_c_id_test = list(set(u_c_id)-set(u_c_id_train))
    return u_c_id_train, u_c_id_test


def x_y_train(u_c_id_train):
    while 1:
        with open('tafeng_json_data_more_transactions_item_id') as f:
            for line in f:
                if json.loads(line).keys()[0] in u_c_id_train:
                    X_Y = list()
                    for i in json.loads(line).values()[0]:
                        list_of_zeros = [0.0] * len(unique_item_id)
                        for j in i:
                            list_of_zeros[j] = 1.0
                        X_Y.append(list_of_zeros)
                    L = len(json.loads(line).values()[0])
                    X = X_Y[0:L - 1]
                    Y = X_Y[1:L]
                    # X = X_Y[-K - 1:-1]
                    # Y = X_Y[-K:]
                    yield (np.array([X]), np.array([Y]))


def x_y_test(u_c_id_test):
    while 1:
        with open('tafeng_json_data_more_transactions_item_id') as f:
            for line in f:
                if json.loads(line).keys()[0] in u_c_id_test:
                    X_Y = list()
                    for i in json.loads(line).values()[0]:
                        list_of_zeros = [0.0] * len(unique_item_id)
                        for j in i:
                            list_of_zeros[j] = 1.0
                        X_Y.append(list_of_zeros)
                    L = len(json.loads(line).values()[0])
                    X = X_Y[0:L - 1]
                    Y = X_Y[1:L]
                    # X = X_Y[-K - 1:-1]
                    # Y = X_Y[-K:]
                    yield np.array([X]), np.array([Y])


def build_model(input_dim, input_length=None, hidden_units=[100, 100], dropout=0.1, output_activation='sigmoid'):
    model = Sequential()
    model.add(LSTM(hidden_units[0], input_shape=(input_length, input_dim), return_sequences=True))
    model.add(Dropout(dropout))
    for h in hidden_units[1:]:
        model.add(LSTM(h, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(TimeDistributed(Dense(input_dim, activation=output_activation)))
    return model


model = build_model(input_dim=len(unique_item_id))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
# model.fit_generator(x_y_train(0.8, K=5), samples_per_epoch=2560, nb_epoch=5)
uid_train, uid_test = u_c_id_for_train_test(0.8)
print(len(uid_train))
print(len(uid_test))
# model.fit_generator(x_y_train(uid_train, K=5), samples_per_epoch=256, nb_epoch=1)
model.fit_generator(x_y_train(uid_train), samples_per_epoch=len(uid_train), nb_epoch=1)
# model.save('my_model.h5')
# model.fit_generator(x_y_train(uid_train), samples_per_epoch=10, nb_epoch=1)
# scores = model.evaluate_generator(x_y_train(uid_train, K=5), 10)
# scores = model.evaluate_generator(x_y_train(uid_train), len(uid_train))
# print(scores)
# print("Accuracy: %.2f%%" % (scores[1]*100))

# model.fit(X, Y, nb_epoch=3, batch_size=64)
# model.fit(example_input, example_output, batch_size=16)
# prediction = model.predict_generator(x_y_test(uid_test, K=5), val_samples=7555)
# prediction = model.predict_generator(x_y_test(uid_test), val_samples=len(uid_test))
G = x_y_test(uid_test)
# G = x_y_test(uid_train)
recall = list()
precision = list()
F1_score = list()
for i in range(len(uid_test)):
    X_te, Y_te = next(G)
    prediction = model.predict(X_te)
    # print(len(prediction[0]))
    # print(len(Y_tr[0]))
    print(prediction[0])
    print(Y_te[0])
    for j in range(len(prediction[0])):
        print(max(prediction[0][j]))
        est_basket = [i for i, v in enumerate(prediction[0][j]) if v > 0.25]
        true_basket = [i for i, v in enumerate(Y_te[0][j]) if v > 0.25]
        r = len([val for val in est_basket if val in true_basket]) / len(true_basket)
        recall.append(r)
        p = len([val for val in est_basket if val in true_basket]) / len(est_basket)
        precision.append(p)
        F1 = (2.0 * r * p) / (r + p)
        F1_score.append(F1)

